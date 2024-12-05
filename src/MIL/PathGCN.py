import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class PatchGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.5):
        super(PatchGCN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))

        self.layers.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.attention_pooling = nn.Linear(out_channels, 1)  # Match out_channels

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        attention_weights = torch.softmax(self.attention_pooling(x), dim=0)
        global_feature = torch.sum(attention_weights * x, dim=0, keepdim=True)

        return global_feature, attention_weights
    
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _,_, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        _,Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

# Graph construction helper function
def construct_graph(features, adjacency_matrix):
    edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).T
    return Data(x=features, edge_index=edge_index)


