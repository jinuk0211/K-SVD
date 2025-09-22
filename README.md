# SVD
---------------------
W(U x sigma x V.T) @ hidden_states 

W -> U x S | V.T 파라미터 개수 감소 

m x n -> r(m + n) 

vram memory 61.06GB -> 37.8GB 

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
# shared-svd
----------------------
share_V = True

: frequency 기반 가장 많이 router가 선택하는 expert의 v matrix를 공유

128개중 하나 선택해 공유

# K-SVD
----------------------
