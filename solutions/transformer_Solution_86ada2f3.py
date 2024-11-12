class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.lin = nn.Linear(config.d_model, config.vocab_size)
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.ctx_len, config.d_model)
        self.L = nn.CrossEntropyLoss()
        self.ctx_len = config.ctx_len
        self.device = config.device
    
    def forward(self, x, targets=None):
        B, T = x.shape
        x_tok = self.wte(x)
        x_pos = self.wpe(torch.arange(T, device=self.device))
        x = x_tok + x_pos # (B, T, C)

        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lin(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # compute xentropy loss, targets are (B, T)
            B, T, V = logits.shape
            targets = targets.view(B*T)
            logits = logits.view(B*T, V)
            loss = self.L(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_len=256):
        for _ in range(max_len):
            idx_window = idx[:, -self.ctx_len:]
            logits, _ = self(idx_window) #(B, T, V)
            logits = logits[:,-1,:]
            prob = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1) # greedy sample
            idx = torch.cat((idx, next_token), dim=1)
        
        return idx
    

# Testing your implementation
config = Config(
    vocab_size=100,
    d_model=256,
    ctx_len=64,
    n_layers=4
)
decoder = Decoder(config)

x = torch.randint(0, 100, (1, 10))
logits, loss = decoder(x, x)

out = decoder.generate(torch.tensor([[1, 2, 3]]), max_len=5)
print(out.shape)  # Should be (1, 8) - original 3 tokens + 5 new ones