import torch
from torch import nn
import pdb

class Shot_block(nn.Module):
    def __init__(self, num_layers,input_dim, hidden_dim, dropout_rate, activation, embedding1,class_weights,device):
        super().__init__()
        self.embedding1 = nn.Embedding(22, embedding1)

        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(activation())

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Dropout(p=dropout_rate))
            self.layers.append(activation())
        
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(self.layers)
        
        self.class_weights = class_weights
        self.device = device
    def forward(self, x):
        x1 = self.embedding1(x[:,0].long())
        x = torch.cat((x1, x[:,1:]), 1)
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        
        return x

    def derive_weights(self, targets):
        # Compute weights based on targets and class_weights
        weights = torch.zeros(targets.shape[0])
        for i in range(targets.shape[0]):  # Assuming targets is of shape (batch_size, num_classes)
            weights[i] = self.class_weights[targets[i,0].item()] 
        return weights

    def compute_loss(self, outputs, targets):
        weights = self.derive_weights(targets)
        ce_loss = nn.BCELoss(weight=weights.to(self.device))
        targets = targets.squeeze()
        outputs = outputs.squeeze()
        loss = ce_loss(outputs, targets.float())
        return loss