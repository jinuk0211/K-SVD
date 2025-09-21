class DynamicSkippingSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model: Qwen3MoeSparseMoeBlock, beta: float):
        super().__init__()
        # assert isinstance(model, Qwen3MoeSparseMoeBlock)
        # assert model.top_k == 2, "Wrapper currently only supports top-2 MoE" #qwen a30은 8
        self.model = model
        self.beta = beta
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.hidden_dim = model.gate.in_features
        self.experts = model.experts
        self.gate = model.gate

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits and top-k routing
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Dynamic skipping: if top-2 < beta * top-1, zero out top-2
        # mask_skip = topk_weights[:, 1] < self.beta * topk_weights[:, 0]
        # topk_weights[mask_skip, 1] = 0
        mask_skip = topk_weights[:, 1:] < self.beta * topk_weights[:, [0]]
        topk_weights[:, 1:][mask_skip] = 0
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states_flat.dtype)

        # Build expert mask (num_experts, top_k, batch * seq)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
        # Apply skip mask to top-2
        expert_mask[:, 1, mask_skip] = 0

        # Prepare output
        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Compute expert outputs
        for expert_idx in range(self.num_experts):
            idx_topk = torch.where(expert_mask[expert_idx])
            if idx_topk[0].numel() == 0:
                continue

            top_x_list, batch_idx = idx_topk
            current_input = hidden_states_flat[batch_idx]
            expert_output = self.experts[expert_idx](current_input) * topk_weights[batch_idx, top_x_list, None]
            final_hidden_states.index_add_(0, batch_idx, expert_output.to(hidden_states_flat.dtype))

        return final_hidden_states.view(batch_size, seq_len, hidden_dim), router_logits

moe_blk = DynamicSkippingSparseMoeBlockWrapper(svd_block, beta=0.9)
