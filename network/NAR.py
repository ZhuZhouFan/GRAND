import torch
import torch.nn as nn

class NAR(nn.Module):
    def __init__(self,
                 feature_dim:int,
                 cuda:bool = True):
        super(NAR, self).__init__()
        self.device = torch.device(
            "cuda" if cuda & torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=0.0, std=0.01)
        nn.init.normal_(self.beta, mean=0.0, std=0.01)
        nn.init.normal_(self.gamma, mean=0.0, std=0.01)
        
    def forward(self, x, adj_mat, r_prev):
        feature_term = torch.matmul(x, self.beta)
        n_i = torch.sum(adj_mat, dim=1) 
        network_effect = torch.matmul(adj_mat, r_prev) 
        network_effect = self.gamma * network_effect / (n_i + 1e-8)
        output = self.alpha + feature_term + network_effect
        
        return output
    
    def get_parameters(self):
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.detach().cpu().numpy(),
            'gamma': self.gamma.item()
        } 