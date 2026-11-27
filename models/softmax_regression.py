
import torch.nn as nn

class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(196, 10)

    def forward(self, x):
        return self.linear(x)
