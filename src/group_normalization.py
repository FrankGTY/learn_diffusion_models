from torch import nn


class PreNorm(nn.Module):
    """
    PreNorm class, which will be used to apply groupnorm before the attention layer
    """
    
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
        
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)