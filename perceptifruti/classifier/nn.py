import torch
import torch.nn as nn


class RipenessClassifier(nn.Module):
    def __init__(self, filename):
        super().__init__()
        self.fc = nn.Linear(10, 2)

        device = 'cuda' if torch.cuda_is_available() else 'cpu'
        self.load_state_dict(filename)
        self.to(device)
        self.eval()

    def forward(self, x):
        return self.fx(x)
