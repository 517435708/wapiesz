import torch.nn as nn
import torch.nn.functional as F

class ImgNet(nn.Module):
    def __init__(self, *, input_dim=512, hidden_dim=1024, output_dim=300):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        for layer in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


class TxtNet(nn.Module):
    def __init__(self, *, input_dim=300, hidden_dim=512, output_dim=300):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        for layer in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
