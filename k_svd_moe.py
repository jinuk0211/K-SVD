from ksvd import *
class KSVDQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        # self.gate_proj
        # self.up_proj
        # self.down_proje

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def merge_mlp_k_svd(
        self,
        layer_names: list[str] = ["gate_proj", "up_proj", "down_proj"],
        threshold=0.9,
        n_atoms=50,
        sparsity=5,
        n_iter=10,
        device="cpu"
    ):
        """
        Apply shared dictionary K-SVD to all projection layers of Qwen3MoeMLP experts.
        """

        # Collect weights and map expert-layer positions
        all_weights = {layer: [] for layer in layer_names}
        expert_map = {layer: [] for layer in layer_names}

        for expert_idx, expert in enumerate(self.experts):
            for layer in layer_names:
                if not hasattr(expert, layer):
                    raise AttributeError(f"Expert {expert_idx} missing layer '{layer}'")
                linear_layer = getattr(expert, layer)
                if not isinstance(linear_layer, nn.Linear):
                    continue  # Already compressed or non-linear layer

                weight = linear_layer.weight.detach().cpu().numpy()
                all_weights[layer].append(weight)
                expert_map[layer].append((expert_idx, linear_layer.weight.shape))

        # Run K-SVD + clustering per layer type
        compressed_results = {}
        for layer in layer_names:
            if len(all_weights[layer]) == 0:
                continue
            print(f"Compressing layer {layer} with {len(all_weights[layer])} experts.")
            # print(f'{all_weights[layer]}')
            # Apply clustering and K-SVD
            results = cluster_and_shared_ksvd(
                all_weights[layer],
                threshold=threshold,
                n_atoms=n_atoms,
                sparsity=sparsity,
                n_iter=n_iter
            )
            compressed_results[layer] = results

            # ---- layer별 D_shared 한 번만 생성 ----
            D_shared = nn.Parameter(torch.tensor(results[0]['D'], dtype=torch.float32, device=device))
            setattr(self, f"D_shared_{layer}", D_shared)

            # 각 expert별 X 하나만 생성
            for expert_idx, (exp_idx, weight_shape) in enumerate(expert_map[layer]):
                # results[0]['X_cluster']에서 expert별 slice 하나만 선택
                X = torch.tensor(results[0]['X_cluster'][:, expert_idx], dtype=torch.float32, device=device)
                shared_linear = SharedDictLinear(D_shared, X, bias=False).to(device)
                setattr(self.experts[exp_idx], layer, shared_linear)

        return compressed_results
