class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.d_model, 4*config.d_model)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4*config.d_model, config.d_model)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
