import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lera.model.moe import MoE, MoEWithLoadBalancing
from lera.model.rope import RoPE


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0, is_causal=True, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.is_causal = is_causal

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
        # Initialize RoPE
        self.rope = RoPE(d_model=dim, n_heads=n_heads, max_seq_len=max_seq_len, device=None)



    def attn(self, x, attn_mask=None):
        B, S, D = x.shape
        head_dim = D // self.n_heads
        num_heads = self.n_heads

        qkv = self.qkv(x)

        q, k, v = qkv.split(D, dim=2)
        q = q.view(B, S, num_heads, head_dim)
        k = k.view(B, S, num_heads, head_dim)
        v = v.view(B, S, num_heads, head_dim)
        
        # Apply RoPE to q and k
        q, k = self.rope(q, k)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, attn_mask=attn_mask)


        attn_out = attn_out.permute(0, 2, 1, 3)
        attn_out = attn_out.contiguous().view(B, S, D)
        attn_out = self.o(attn_out)

        return attn_out

        
    def forward(self, x, attn_mask=None):
        attn_out = self.attn(x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)

        x = self.norm2(x + mlp_out)
        return x

    def __call__(self, x, attn_mask=None):
        return self.forward(x, attn_mask=attn_mask)

class TransformerBlockWithLoadBalancing(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0, is_causal=True, 
                 use_moe=False, num_experts=4, verbose_router=False, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.num_experts = num_experts
        self.dropout = dropout
        self.is_causal = is_causal
        self.verbose_router = verbose_router

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.use_moe = use_moe
        if use_moe:
            self.router = nn.Linear(dim, num_experts)
            self.qkv = MoEWithLoadBalancing(dim, 3 * dim, num_experts, dropout=dropout)
            self.o = MoEWithLoadBalancing(dim, dim, num_experts, dropout=dropout)
            self.mlp_in = MoEWithLoadBalancing(dim, int(dim * mlp_ratio), num_experts, dropout=dropout)
            self.mlp_out = MoEWithLoadBalancing(int(dim * mlp_ratio), dim, num_experts, dropout=dropout)
        else:
            self.qkv = nn.Linear(dim, 3 * dim)
            self.o = nn.Linear(dim, dim)
            self.mlp_in = nn.Linear(dim, int(dim * mlp_ratio))
            self.mlp_out = nn.Linear(int(dim * mlp_ratio), dim)
        
        # Initialize RoPE
        self.rope = RoPE(d_model=dim, n_heads=n_heads, max_seq_len=max_seq_len, device=None)

    def forward(self, x, router_vector, attn_mask=None, return_load_balance_loss=False):
        B, S, D = x.shape
        total_lb_loss = 0.0
        
        # Attention block
        if self.use_moe:
            router_out = self.router(router_vector)
            expert_probs = F.softmax(router_out, dim=-1)
            
            if self.verbose_router:
                top_experts = torch.argmax(expert_probs, dim=-1)
                counts = torch.bincount(top_experts.flatten(), minlength=self.num_experts)
                usage = counts.float() / counts.sum() * 100
                print(f"Expert usage (%): {[f'{u:.1f}' for u in usage.tolist()]}")
            
            # QKV projection
            if return_load_balance_loss:
                qkv, _, lb_loss_qkv = self.qkv(x, expert_probs, return_load_balance_loss=True)
                total_lb_loss += lb_loss_qkv
            else:
                qkv, _ = self.qkv(x, expert_probs)
        else:
            qkv = self.qkv(x)

        # Split into Q, K, V and compute attention
        q, k, v = qkv.split(D, dim=2)
        q = q.view(B, S, self.n_heads, D // self.n_heads)
        k = k.view(B, S, self.n_heads, D // self.n_heads)
        v = v.view(B, S, self.n_heads, D // self.n_heads)
        
        # Apply RoPE to q and k
        q, k = self.rope(q, k)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, attn_mask=attn_mask)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        
        # Output projection
        if self.use_moe:
            if return_load_balance_loss:
                attn_out, _, lb_loss_o = self.o(attn_out, expert_probs, return_load_balance_loss=True)
                total_lb_loss += lb_loss_o
            else:
                attn_out, _ = self.o(attn_out, expert_probs)
        else:
            attn_out = self.o(attn_out)
        
        x = self.norm1(x + attn_out)
        
        # MLP block
        if self.use_moe:
            if return_load_balance_loss:
                mlp_out, _, lb_loss_mlp_in = self.mlp_in(x, expert_probs, return_load_balance_loss=True)
                total_lb_loss += lb_loss_mlp_in
            else:
                mlp_out, _ = self.mlp_in(x, expert_probs)
            
            mlp_out = F.gelu(mlp_out)
            
            if return_load_balance_loss:
                mlp_out, _, lb_loss_mlp_out = self.mlp_out(mlp_out, expert_probs, return_load_balance_loss=True)
                total_lb_loss += lb_loss_mlp_out
            else:
                mlp_out, _ = self.mlp_out(mlp_out, expert_probs)
        else:
            mlp_out = F.gelu(self.mlp_in(x))
            mlp_out = self.mlp_out(mlp_out)
        
        x = self.norm2(x + mlp_out)
        
        if return_load_balance_loss:
            return x, total_lb_loss
        return x



