class SingleHeadCausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.d_model // config.n_heads
        self.key = nn.Linear(config.d_model, self.head_dim, bias=False)
        self.query = nn.Linear(config.d_model, self.head_dim, bias=False)
        self.values = nn.Linear(config.d_model, self.head_dim, bias=False)

        self.register_buffer("cmask", torch.tril(torch.ones([config.ctx_len, config.ctx_len])))

    
    def forward(self, x):

        B, T, C = x.shape
        
        K = self.key(x) # (B, T, C) @ (_, C, H) -> (B, T, H)
        Q = self.query(x)
        V = self.values(x)

        y = Q @ K.transpose(-2, -1) * self.head_dim**-0.5 # (B, T, H) @ (B, H, T) -> (B, T, T)
        y = torch.masked_fill(y, self.cmask[:T, :T]==0, float('-inf'))
        y = F.softmax(y, dim=-1) @ V
        return y
    

# Test your implementation
config = Config(d_model=256, n_heads=8, ctx_len=16)
attention = SingleHeadCausalAttention(config)
x = torch.randn(2, 10, 256)  # (batch_size, seq_len, d_model)
output = attention(x)
assert output.shape == (2, 10, 32)  # head_dim = 256/8 = 32