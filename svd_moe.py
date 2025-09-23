
class moe_layer(nn.Module):
  def __init__(self, config, share_v, shared_vgate, shared_vup, shared_vdown ):
      super().__init__()

      self.intermediate_dim = config.moe_intermediate_size #768
      self.hidden_dim = config.hidden_size #2048
      self.dtype = torch.bfloat16
      self.low_rank = int(self.intermediate_dim * self.hidden_dim * 0.3 / (self.intermediate_dim + self.hidden_dim))
      # self.delta_ratio = 0.3
      self.act_fn = F.relu #ACT2FN[config.hidden_act]
      # self.low_rank =
      if share_v == True:
        self.v1 = shared_vgate
        # self.experts_v1_shared_gate = nn.Linear(self.hidden_dim, self.low_rank, bias=False, dtype=torch.bfloat16)         
        self.us1 = nn.Linear(self.low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v2 = shared_vup
        self.us2 = nn.Linear(self.low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v3 = shared_vdown
        self.us3 = nn.Linear(self.low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
      else:
        self.v1 = nn.Linear(self.hidden_dim, self.low_rank, bias=False, dtype=torch.bfloat16)
        self.us1 = nn.Linear(self.low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v2 = nn.Linear(self.hidden_dim, self.low_rank, bias=False, dtype=torch.bfloat16)
        self.us2 = nn.Linear(self.low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v3 = nn.Linear(self.intermediate_dim, self.low_rank, bias=False, dtype=torch.bfloat16)
        self.us3 = nn.Linear(self.low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
      # self.low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))

  def forward(self,hidden_states):
      hidden_states = hidden_states.to(self.v1.weight.dtype)
      gate = self.us1(self.v1(hidden_states))
      up = self.us2(self.v2(hidden_states))
      x = self.us3(self.v3(self.act_fn(gate) * up))
      return x


class SVDQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config,dtype= torch.bfloat16):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # gating
        self.dtype = dtype
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=self.dtype)
        # share v
        self.low_rank = int(self.intermediate_dim * self.hidden_dim * 0.3 / (self.intermediate_dim + self.hidden_dim))
        self.experts_v1_shared_gate = nn.Linear(self.hidden_dim, self.low_rank, bias=False, dtype=self.dtype)
        self.experts_v2_shared_up = nn.Linear(self.hidden_dim, self.low_rank, bias=False, dtype=self.dtype)
        self.experts_v3_shared_down = nn.Linear(self.intermediate_dim, self.low_rank, bias=False, dtype=self.dtype)
        self.beta = 0.9
        self.experts = nn.ModuleList(
            [moe_layer(config, False, self.experts_v1_shared_gate, self.experts_v2_shared_up, self.experts_v3_shared_down)  for _ in range(self.num_experts)])
        # share v false
        # self.experts = nn.ModuleList(
        #     [moe_layer(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        # )
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

    def SVD(self, modules, num_experts=128, dtype=torch.bfloat16):
        self.gate.weight = modules.gate.weight
        
        # Process gate_proj weights
        for i in range(num_experts):
            if i % 10 == 0:
                print(f'Processing gate_proj for expert {i}/{num_experts}')
            
            # Keep original weight on GPU, but move float copy to CPU for SVD if still OOM
            # For now, let's keep it on GPU but manage carefully
            original_weight = modules.experts[i].gate_proj.weight
            
            # 1. Upcast and perform SVD
            u, s, v = torch.svd_lowrank(original_weight.float(), q=self.low_rank)
            
            # 2. Create the first low-rank matrix, assign it, and immediately delete v
            self.experts[i].v1.weight = nn.Parameter(v.T.to(dtype))
            del v
            
            # 3. Create the second low-rank matrix, assign it, and immediately delete u and s
            US_top = u @ torch.diag(s)
            self.experts[i].us1.weight = nn.Parameter(US_top.to(dtype))
            del u, s, US_top
            
            if i % 20 == 0:
                torch.cuda.empty_cache()

        # Process up_proj weights
        for i in range(num_experts):
            if i % 10 == 0:
                print(f'Processing up_proj for expert {i}/{num_experts}')
            
            original_weight = modules.experts[i].up_proj.weight
            u, s, v = torch.svd_lowrank(original_weight.float(), q=self.low_rank)
            
            self.experts[i].v2.weight = nn.Parameter(v.T.to(dtype))
            del v
            
            US_top = u @ torch.diag(s)
            self.experts[i].us2.weight = nn.Parameter(US_top.to(dtype))
            del u, s, US_top

            if i % 20 == 0:
                torch.cuda.empty_cache()

        # Process down_proj weights
        for i in range(num_experts):
            if i % 10 == 0:
                print(f'Processing down_proj for expert {i}/{num_experts}')
            
            original_weight = modules.experts[i].down_proj.weight
            u, s, v = torch.svd_lowrank(original_weight.float(), q=self.low_rank)
            
            self.experts[i].v3.weight = nn.Parameter(v.T.to(dtype))
            del v
            
            US_top = u @ torch.diag(s)
            self.experts[i].us3.weight = nn.Parameter(US_top.to(dtype))
            del u, s, US_top

            if i % 20 == 0:
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()            
        # self.shared_u = v_list.average()
        # self.shared_u = torch.stack(v_list, dim=0).mean(dim=0)

