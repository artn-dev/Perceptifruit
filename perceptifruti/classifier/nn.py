import torch
import torch.nn as nn


class RipenessClassifier(nn.Module):
    def __init__(self, filename):
        super().__init__()
        self.fc = nn.Linear(10, 2)

        
        device = 'cpu'

        
        self.to(device)
        self.eval()

    def forward(self, x):
        return self.fx(x)
