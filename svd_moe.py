
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
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.to(self.gate.weight.dtype)
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Compute router logits and routing weights
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Dynamic skipping: if any expert weight < beta * top_expert_weight, skip it
        if hasattr(self, 'beta') and self.beta > 0:
            # Create mask for experts to skip (all except top-1)
            mask_skip = routing_weights[:, 1:] < self.beta * routing_weights[:, [0]]
            # Zero out weights for skipped experts
            routing_weights[:, 1:][mask_skip] = 0
        
        # Normalize routing weights (including after dynamic skipping)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Prepare output tensor
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        
        # Build expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # Apply dynamic skip mask to expert_mask
        if hasattr(self, 'beta') and self.beta > 0:
            # Zero out expert assignments for dynamically skipped experts
            for k_idx in range(1, self.top_k):
                expert_mask[:, k_idx, mask_skip[:, k_idx-1]] = 0
        
        # Find which experts are actually used (optimization for 128 experts)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
        
        # Process each active expert
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.numel() == 0:  # Skip if no tokens assigned to this expert
                continue
                
            # Get input tokens for this expert
            current_state = hidden_states[top_x].reshape(-1, hidden_dim)
            
            # Apply expert and weight by routing weights
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            
            # Accumulate results
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # Reshape back to original dimensions
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits

    def SVD(self, modules, top_k_svd = 40,num_experts = 128):
        self.gate.weight = modules.gate.weight
        v_list = []
        for i in range(num_experts):
          if i % 10 == 0:
            print(f'gate_proj의 {i}/128번째 expert svd')
              
          u,s,v = torch.svd_lowrank(modules.experts[i].gate_proj.weight.float(), q=self.experts[0].low_rank)
          # v_list.append(v)
          # u_list의 평균값 구하기
          self.experts[i].v1.weight = nn.Parameter(v.T)
          # US_top = u[:, :top_k_svd] * torch.diag(s[:top_k_svd])
          US_top = u @ torch.diag(s)
          self.experts[i].us1.weight = nn.Parameter(US_top)
          del u
          del s
          del v
        # self.shared_u = v_list.average()
          if i % 20 == 0:
            torch.cuda.empty_cache()
        # self.shared_u = torch.stack(v_list, dim=0).mean(dim=0)


        v_list = []
        for i in range(num_experts):
          if i % 10 == 0:
            print(f'up proj의 {i}/128번째 svd')
          u,s,v = torch.svd_lowrank(modules.experts[i].up_proj.weight.float(), q=self.experts[0].low_rank)
          # v_list.append(v)
          self.experts[i].v2.weight = nn.Parameter(v.T)
          # US_top = u[:, :top_k_svd] * torch.diag(s[:top_k_svd])
          US_top = u @ torch.diag(s)
          self.experts[i].us2.weight = nn.Parameter(US_top)
          del u
          del s
          del v
          if i % 20 == 0:
            torch.cuda.empty_cache()            
        # self.shared_u = v_list.average()
        # self.shared_u = torch.stack(v_list, dim=0).mean(dim=0)


        v_list = []
        for i in range(num_experts):
          if i % 10 == 0:
            print(f'down_proj의 {i}/128번째 svd')
          u,s,v = torch.svd_lowrank(modules.experts[i].down_proj.weight.float(), q=self.experts[0].low_rank)
          # v_list.append(v)
          self.experts[i].v3.weight = nn.Parameter(v.T)
          # US_top = u[:, :top_k_svd] * torch.diag(s[:top_k_svd])
          US_top = u @ torch.diag(s)
          self.experts[i].us3.weight = nn.Parameter(US_top)
          del u
          del s
          del v
          if i % 20 == 0:
            torch.cuda.empty_cache()            
        # self.shared_u = v_list.average()
        # self.shared_u = torch.stack(v_list, dim=0).mean(dim=0)

