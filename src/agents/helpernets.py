import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        
        super().__init__()
        self.Q = nn.Parameter(torch.randn(embed_dim))   # learnable query vector [d]
        self.K = nn.Linear(embed_dim, embed_dim)        # maps input -> keys
        self.V = nn.Linear(embed_dim, embed_dim)        # maps input -> values

    def forward(self, h_inputs: torch.Tensor):
        
        keys = self.K(h_inputs)        # [N, d]
        values = self.V(h_inputs)      # [N, d]

        eij = torch.matmul(keys, self.Q)    # [N, d] · [d] -> [N]
        alphas = torch.softmax(eij, dim=0)  # [N]

        h_out = torch.sum(values * alphas.unsqueeze(-1), dim=0)  # [d]
        return h_out, alphas
