import torch as T
import torch.nn as nn

class ImgNet(nn.Module):
    def __init__(self, *, input_dim=512, hidden_dim=1024, output_dim=300):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        for layer in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x

class TxtNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
