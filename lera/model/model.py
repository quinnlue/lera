import torch
import torch.nn as nn
from lera.model.transformer import TransformerBlockWithLoadBalancing, TransformerBlock
import torch.nn.functional as F

class RRMWithLoadBalancing(nn.Module):
    def __init__(self, d_model, n_heads, max_recursions, vocab_size, num_experts, 
                 context_len, preprocess_depth, temperature_idx, fact_vector_idx, router_idx):
        super().__init__()
        self.max_recursions = max_recursions
        self.context_len = context_len
        self.tau = 0.5

        self.router_idx = router_idx
        self.temperature_idx = temperature_idx
        self.fact_vector_idx = fact_vector_idx

    
        self.encoder = nn.Embedding(vocab_size, d_model)
        
        self.recursive_block = TransformerBlockWithLoadBalancing(
            d_model, n_heads, is_causal=False,
            use_moe=True, num_experts=num_experts, 
            verbose_router=False, max_seq_len=context_len
        )

        self.preprocess_block = nn.ModuleList([
            TransformerBlock(d_model, n_heads, is_causal=False, max_seq_len=context_len) for _ in range(preprocess_depth)
        ])

        self.temperature_projection = nn.Linear(d_model, 1)

        self.gate = nn.Linear(d_model, d_model)

        self.decoder = nn.Linear(d_model, vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, x, context_length, attn_mask=None, context_mask=None, y_true=False, return_load_balance_loss=False, return_deep_supervision_loss=False):
        X_B, X_S, X_D = x.shape

        total_lb_loss = 0.0
        total_deep_supervision_loss = 0.0
        
        # Initialize context from input
        context = x[:, :self.context_len, :]

        # Preprocess the context
        for block in self.preprocess_block:
            context = block(context, attn_mask=attn_mask)
        
        # Recursive processing
        for _ in range(self.max_recursions):
            temperature_vector = context[:, self.temperature_idx, :]
            fact_vector = context[:, self.fact_vector_idx, :]

    
            if return_load_balance_loss:
                router_vector = context[:, self.router_idx, :]  # Use context router, not original input
                update, lb_loss = self.recursive_block(context, router_vector, attn_mask=attn_mask, return_load_balance_loss=True)
                
                # Compute gate from context (not original x)
                gate = torch.sigmoid(self.gate(context))
                temperature = torch.sigmoid(self.temperature_projection(temperature_vector)).unsqueeze(1)

                # Update context
                context = context + (gate * update * temperature)

                # Deep supervision: predict at each recursion step
                if return_deep_supervision_loss:
                    fact_vector = context[:, self.fact_vector_idx, :]
                    predicted = self.decoder(fact_vector)

                    deep_supervision_loss = F.cross_entropy(
                        input=predicted,
                        target=y_true,
                        reduction="mean"
                    )
                    total_deep_supervision_loss += deep_supervision_loss

                total_lb_loss += lb_loss

        
        # Final output: use the recursively updated context, not original input!
        fact_vector = context[:, self.fact_vector_idx, :]
        x = self.decoder(fact_vector)

        if return_load_balance_loss:
            return x, total_lb_loss, total_deep_supervision_loss
        else:
            return x