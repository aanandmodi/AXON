import torch
import torch.nn as nn

class HandGestureMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) for custom hand gesture classification from 63-dimensional landmarks."""
    def __init__(self, num_classes: int):
        super(HandGestureMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def __repr__(self) -> str:
        return f"HandGestureMLP(layers=[63, 128, 64, output])"
