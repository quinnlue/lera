import torch
import torch.nn as nn
from lera.model.transformer import TransformerBlock



class Model(nn.Module):
    def __init__(self, d_model, n_heads, vocab_size, output_dim, max_seq_len, depth):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, is_causal=False, max_seq_len=max_seq_len)
            for _ in range(depth)
        ])
        self.decoder = nn.Linear(d_model, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, x, attention_mask=None):
        x = self.encoder(x)
        
        # Convert attention_mask to format expected by scaled_dot_product_attention
        # Input mask: (batch_size, seq_len) with 1=attend, 0=ignore
        # Output mask: (batch_size, 1, 1, seq_len) with True=attend, False=ignore
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.bool()  # Convert to boolean
            # Expand to (batch_size, 1, 1, seq_len) for broadcasting across heads and queries
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask=attn_mask)
        return self.decoder(x[:, -1, :])