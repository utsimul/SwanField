import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class attention_pool(nn.Module):

    def __init__(
            self, 
            num_assets : int,
            asset_dim : int,
            embed_dim : int,

    ):
        
        super().__init__()
        self.Q = nn.Parameter(torch.randn(embed_dim)) #d cross 1
        self.K = nn.Linear(embed_dim, embed_dim) # d cross d
        self.V = nn.Linear(embed_dim, embed_dim) # d cross d

    def forward(self, h_assets):

        keys = self.K(h_assets) #n cross d ----------> can add batch cross n cross d but functionally n cross d
        values = self.V(h_assets) #n cross d

        eij = torch.matmul(keys, self.query) #n cross 1
        alphas = torch.softmax(eij, dim=1)  #n cross 1

        h_sector = torch.sum(values * alphas.unsqueeze(-1), dim=1) #d cross 1
        return h_sector, alphas