import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, mlp_ratio=4, mlp_p=0):
        super().__init__()
        self.fc_1 = nn.Linear(in_channels, in_channels * mlp_ratio)
        self.act = nn.GELU()
        self.drop_1 = nn.Dropout(mlp_p)
        self.fc_2 = nn.Linear(in_channels * mlp_ratio, in_channels)
        self.drop_2 = nn.Dropout(mlp_p)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.act(x)
        x = self.drop_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        return x
