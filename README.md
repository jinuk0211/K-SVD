# SVD decompositoin
=============================
W(U x sigma x V.T) @ hidden_states 

W -> U x S | V.T 파라미터 개수 감소 

m x n -> r(m + n) 

vram memory 사용량 61.06GB -> 37.8GB, 20.42GB(rank = 167)
#--------------------------------------------
<img width="899" height="456" alt="image" src="https://github.com/user-attachments/assets/6095d026-e5f6-46d0-b896-4d5b8e22488e" />
<img width="1171" height="367" alt="image" src="https://github.com/user-attachments/assets/40c18d90-832f-4bb9-be35-a646d2fa36c4" />

A100 PCle -> A40, RTX A6000

```python
svd.ipynb 파일 일부를 작성
from transformers import AutoConfig, AutoModelForCausalLM
from svd_moe import SVDQwen3MoeSparseMoeBlock

model_name = "Qwen/Qwen3-30B-A3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
    if i % 5 == 0:
        before_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Layer {i} - Before: {before_mem:.2f}GB")
    
    # 기존 MLP 백업
    old_mlp = model.model.layers[i].mlp
    device = old_mlp.gate.weight.device
    
    # 새 MoE 블록 생성
    Merge_MoE_Block = SVDQwen3MoeSparseMoeBlock(model.config).to(device)
    Merge_MoE_Block.SVD(old_mlp)
    
    # 교체 전 기존 MLP 명시적 삭제
    model.model.layers[i].mlp = None  # 참조 끊기
    del old_mlp                       # 명시적 삭제
    torch.cuda.empty_cache()          # 캐시 정리
    
    # 새 MLP 할당
    model.model.layers[i].mlp = Merge_MoE_Block
    del Merge_MoE_Block
    
    # 메모리 정리
    torch.cuda.empty_cache()
    
    if i % 5 == 0:
        after_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Layer {i} - After: {after_mem:.2f}GB, Saved: {before_mem-after_mem:.2f}GB")

```
#dynamic skipping
----------------------
```python
# SVDQwen3MoeSparseMoeblock의 forward를 아래 버젼으로 바꾸기
# self.beta 값으로 pruning할 expert 정도 조절 가능

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        print(f'hidden_states: {hidden_states.shape}')
        hidden_states = hidden_states.view(-1, hidden_dim)
        print(f'hidden_states: {hidden_states.shape}')
        
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        print(f'top k 전 routing_weights shape: {routing_weights.shape}')
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        print(f'top k 후 routing_weights: {routing_weights.shape}{routing_weights}')
        print(f'selected_experts: {selected_experts}')
        
        #------------------------------------------------------
        if self.top_k > 1:
            top1_weights = routing_weights[:, 0:1]
            print(f'top1_weights: {top1_weights.shape}{top1_weights}')

            other_weights = routing_weights[:, 1:]
            print(f'other_weights: {other_weights.shape}{other_weights}')

            # Shape: (num_tokens, top_k - 1)
            pruning_mask = other_weights < self.beta * top1_weights
            print(f'pruning_mask: {pruning_mask.shape} {pruning_mask}')
            
            # Set the weights of the pruned experts to 0
            routing_weights[:, 1:].masked_fill_(pruning_mask, 0)   
        #------------------------------------------------------     
        
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        print(f'expert_mask: {expert_mask.shape}, {expert_mask}')
        
        #------------------------------------------------------
        if self.top_k > 1:
            # expert_mask shape: (num_experts, top_k, num_tokens)  
            # pruning_mask shape: (num_tokens, top_k - 1) - True where we want to prune
            # expert_mask[:, 1:, :] shape: (num_experts, top_k-1, num_tokens)
            
            # Transpose pruning_mask to (top_k-1, num_tokens) and add expert dimension
            pruning_mask_expanded = pruning_mask.t().unsqueeze(0)  # Shape: (1, top_k-1, num_tokens)
            
            # Use masked_fill_ to zero out pruned positions
            # We want to set to 0 where pruning_mask_expanded is True
            expert_mask[:, 1:, :].masked_fill_(pruning_mask_expanded, 0)
        #------------------------------------------------------
        
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        print(f'expert_hit: {expert_hit}')
        
        for expert_idx in expert_hit:
            print(f'expert_idx: {expert_idx}')
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits        
```
# shared-svd
----------------------
share_V = True

: frequency 기반 가장 많이 router가 선택하는 expert의 v matrix를 공유

128개중 하나 선택해 공유

# K-SVD
----------------------
