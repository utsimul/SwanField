import torch
import torch.nn as nn
import torch.nn.functional as F

class attention_pool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(embed_dim))   # [d]
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)

    def forward(self, h_inputs: torch.Tensor):
        """
        h_inputs: [batch, N, d]
        Returns:
            h_out: [batch, d]
            alphas: [batch, N]
        """
        keys = self.K(h_inputs)      # [batch, N, d]
        values = self.V(h_inputs)    # [batch, N, d]

        # Compute attention scores
        # Q: [d] → broadcast to [batch, d]
        eij = torch.matmul(keys, self.Q)      # [batch, N, d] · [d] -> [batch, N]

        alphas = torch.softmax(eij, dim=1)    # softmax over assets per batch

        h_out = torch.sum(values * alphas.unsqueeze(-1), dim=1)  # [batch, d]
        print(GREEN + "h_out from attention pool is: ", h_out)

        return h_out, alphas
