import torch
import torch.nn as nn


# MLP Model with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_dim, hidden, seed=None):
        super(MLP, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        if isinstance(input_dim, tuple):
            # (seq_len, feature_dim)
            self.seq_len = input_dim[0]
            self.feature_dim = input_dim[1]
            flattened_dim = self.seq_len * self.feature_dim
        else:
            flattened_dim = input_dim
            
        self.flatten = nn.Flatten() 
        self.network = nn.Sequential(
            nn.Linear(flattened_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//2, 1),
        )
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        if len(x.shape) == 3:
            # [batch_size, seq_len, feature_dim]
            x = self.flatten(x)  # [batch_size, seq_len * feature_dim]

        logits = self.network(x)
        preds = self.activation(logits)

        return {
            'logits': logits,
            'preds': preds
        }
