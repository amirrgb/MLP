import torch
import torch.nn as nn


# ---------- Perceptron ----------

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # raw score (no sigmoid here)


# ---------- MLP ----------

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=1, output_dim=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    

class MLP_MultiClass(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.model(x)
    

class MLP_MultiClass_BN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.model(x)