import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size: int = 12, hidden_size: int = 128, output_size: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNConv(nn.Module):
    def __init__(self, in_channels: int = 3, grid_size: int = 20, output_size: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.fc_size = 64 * grid_size * grid_size
        self.fc1 = nn.Linear(self.fc_size, 256)
        self.fc2 = nn.Linear(256, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_model(model_type: str = "fc", **kwargs) -> nn.Module:
    if model_type == "fc":
        fc_kwargs = {k: v for k, v in kwargs.items() if k in ['input_size', 'hidden_size', 'output_size']}
        return DQN(**fc_kwargs)
    elif model_type == "conv":
        conv_kwargs = {k: v for k, v in kwargs.items() if k in ['in_channels', 'grid_size', 'output_size']}
        return DQNConv(**conv_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")