import torch
import torch.nn as nn
import torch.nn.functional as F
from k_svd_moe import KSVDQwen3MoeSparseMoeBlock
# --------------------------
# 간단한 Qwen3MoeMLP 정의 (토이)
# --------------------------
class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)

    def forward(self, x):
        return F.relu(self.fc1(x)) @ self.fc2.weight.T  # weight를 사용하는 간단 예시

class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.relu #ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class SharedDictLinear(nn.Module):
    def __init__(self, D, X, bias=False):
        super().__init__()
        self.D = D   # nn.Parameter가 아닌 참조
        self.X = nn.Parameter(X.clone())
        if bias:
            self.bias = nn.Parameter(torch.zeros(D.shape[0]))
        else:
            self.bias = None

    @property
    def weight(self):
        # 기존 nn.Linear와 호환되도록 가상 weight 속성 제공
        return self.D @ self.X

    def forward(self, x):
        out = x @ (self.D @ self.X).T
        if self.bias is not None:
            out = out + self.bias
        return out

# --------------------------
# Config 토이
# --------------------------
class Config:
    hidden_size = 8
    num_experts = 3
    num_experts_per_tok = 2
    norm_topk_prob = True
    moe_intermediate_size = 16

config = Config()


moe_block = KSVDQwen3MoeSparseMoeBlock(config)

def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

# moe_block 파라미터 개수 출력
total_params, trainable_params = count_parameters(moe_block)
print(f"Total parameters in moe_block: {total_params}")
print(f"Trainable parameters in moe_block: {trainable_params}")

# 각 파라미터 이름과 개수도 확인
for name, param in moe_block.named_parameters():
    print(f"{name}: {param.numel()}")

cr, r = moe_block.merge_mlp_k_svd(device="cpu")

moe_blk = moe_block.merge_mlp_k_svd(device="cpu")
total_params, trainable_params = count_parameters(moe_blk)
print(f"Total parameters in moe_block: {total_params}")
print(f"Trainable parameters in moe_block: {trainable_params}")

for name, param in moe_block.named_parameters():
    print(f"{name}: {param.numel()}")

