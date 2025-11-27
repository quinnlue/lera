import torch

class RoPE(torch.nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=4096, base=10000, device=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")

        inv_freq = 1.0 / (base ** (torch.arange(0, self.d_head, 2).float() / self.d_head)).to(self.device)
        self.register_buffer("inv_freq", inv_freq)

        cos, sin = self._build_freqs()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _build_freqs(self):
        t = torch.arange(self.max_seq_len, device=self.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[:, None, None, :]
        sin = emb.sin()[:, None, None, :]
        cos = cos.reshape(1, self.max_seq_len, 1, self.d_head)
        sin = sin.reshape(1, self.max_seq_len, 1, self.d_head)
        return cos, sin

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k):
        B, S, H, D = q.shape
        if q.shape != k.shape:
            raise NotImplementedError("q and k must have the same shape")
        cos, sin = self.cos[:, :S, :, :], self.sin[:, :S, :, :]
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin)

    def __call__(self, q, k):
        return self._apply_rotary_pos_emb(q, k)

if __name__ == "__main__":
    r = RoPE(d_model=512, n_heads=8, device=torch.device("cuda"))
    q = torch.randn(5, 7, 8, 64, device="cuda")
    q_rot, k_rot = r._apply_rotary_pos_emb(q, q)
    print(q_rot.shape, k_rot.shape)
