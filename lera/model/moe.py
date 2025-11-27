import torch
import torch.nn as nn
import math

class LowRankExpert(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Linear(rank, out_dim, bias=False)
        self.B = nn.Linear(in_dim, rank, bias=False)

    def forward(self, x):
        return self.A(self.B(x))

class MoE(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, dropout=0.0, top_k=1):
        super().__init__()
        if top_k != 1:
            raise NotImplementedError("Top-k is not implemented for MoE")
        self.top_k = top_k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        
        # Vectorized parameters
        self.expert_weights = nn.Parameter(torch.empty(num_experts, out_dim, in_dim))
        self.expert_biases = nn.Parameter(torch.empty(num_experts, out_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.expert_weights[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.expert_weights[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.expert_biases[i], -bound, bound)

    def forward(self, x, expert_probs, inference=False):
        if inference:
            raise NotImplementedError("Inference is not implemented for MoE")

        B, S, D = x.shape

        if expert_probs.shape != (B, self.num_experts):
            raise ValueError(f"Expert probabilities must be of shape (B, num_experts), got {expert_probs.shape}")
        
        expert_idx = torch.argmax(expert_probs, dim=-1)

        # Gather weights for the batch
        # w: (B, out_dim, in_dim)
        w = self.expert_weights[expert_idx]
        # b: (B, out_dim)
        b = self.expert_biases[expert_idx]

        # x: (B, S, in_dim)
        # w.transpose(1, 2): (B, in_dim, out_dim)
        # Result: (B, S, out_dim)
        x_out = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)

        return x_out, expert_idx


    def __call__(self, x, expert_probs, inference=False):
        return self.forward(x, expert_probs, inference)

class MoEWithLoadBalancing(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, rank=None, dropout=0.0, top_k=1):
        super().__init__()
        if top_k != 1:
            raise NotImplementedError("Top-k is not implemented for MoE")
        self.top_k = top_k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.rank = rank
        
        if self.rank is not None:
            # LowRankExpert: A(B(x)) where B is (in, rank) and A is (rank, out)
            # Weights: A -> (E, out, rank), B -> (E, rank, in)
            self.expert_weights_A = nn.Parameter(torch.empty(num_experts, out_dim, rank))
            self.expert_weights_B = nn.Parameter(torch.empty(num_experts, rank, in_dim))
        else:
            self.expert_weights = nn.Parameter(torch.empty(num_experts, out_dim, in_dim))
            self.expert_biases = nn.Parameter(torch.empty(num_experts, out_dim))
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.rank is not None:
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(self.expert_weights_A[i], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.expert_weights_B[i], a=math.sqrt(5))
        else:
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(self.expert_weights[i], a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.expert_weights[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.expert_biases[i], -bound, bound)

    def compute_load_balancing_loss(self, router_probs, expert_indices, num_experts):
        batch_size = router_probs.shape[0]
        
        # f_i: fraction of tokens routed to expert i
        # Count how many tokens are assigned to each expert
        expert_counts = torch.zeros(num_experts, device=router_probs.device)
        for i in range(num_experts):
            expert_counts[i] = (expert_indices == i).float().sum()
        f_i = expert_counts / batch_size  # [num_experts]
        
        # P_i: mean router probability for expert i across all tokens
        P_i = router_probs.mean(dim=0)  # [num_experts]
        
        # Load balancing loss: N * sum(f_i * P_i)
        # This is minimized when experts are used uniformly
        load_balance_loss = num_experts * (f_i * P_i).sum()
        
        return load_balance_loss

    def forward(self, x, expert_probs, return_load_balance_loss=False):
        B, S, D = x.shape

        if expert_probs.shape != (B, self.num_experts):
            raise ValueError(f"Expert probabilities must be of shape (B, num_experts), got {expert_probs.shape}")
        
        expert_idx = torch.argmax(expert_probs, dim=-1)

        if self.rank is not None:
            # Gather weights
            w_a = self.expert_weights_A[expert_idx] # (B, out, rank)
            w_b = self.expert_weights_B[expert_idx] # (B, rank, in)
            
            # x: (B, S, in)
            # x @ B^T -> (B, S, rank)
            # w_b^T: (B, in, rank)
            intermediate = torch.bmm(x, w_b.transpose(1, 2))
            
            # intermediate @ A^T -> (B, S, out)
            # w_a^T: (B, rank, out)
            x_out = torch.bmm(intermediate, w_a.transpose(1, 2))
            
        else:
            w = self.expert_weights[expert_idx] # (B, out, in)
            b = self.expert_biases[expert_idx]  # (B, out)
            
            x_out = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)

        if return_load_balance_loss:
            lb_loss = self.compute_load_balancing_loss(expert_probs, expert_idx, self.num_experts)
            return x_out, expert_idx, lb_loss
        
        return x_out, expert_idx

if __name__ == "__main__":
    model = MoE(10, 10, 3)
    x = torch.randn(10, 10, 10)
    expert_probs = torch.randn(10, 3)
    x_out, expert_idx = model(x, expert_probs)
    print(x_out.shape)
    print(expert_idx.shape)
