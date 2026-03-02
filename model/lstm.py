from unicodedata import bidirectional
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class LSTMClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_classes, device,  sigmoid=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
        self.device=device
    
    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim)
        c0 = torch.zeros(x.size(0), self.hidden_dim)
        return h0.to(self.device),c0.to(self.device)

    def forward(self, x, return_activation = False):
        x = x.to(self.device)
        hidden,cell = self.init_hidden(x)
                        
        time_steps=x.shape[1]              #shape of x is (batches,time_Steps,features)
        
        for i in range(0,time_steps):
            inputs=x[:,i]                  # (batch,features) shape
            hidden,cell = self.rnn(inputs,(hidden,cell))
        
        logits = self.fc(hidden)              # take the hidden vector corresponding to last time step

        preds = self.activation(logits)

        return {
            "logits": logits,
            "preds": preds,
        }



