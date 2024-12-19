#from os.path import join
#from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from src.MIL.ABMIL import Att_Net_Gated_Dual
from src.MIL.WSI_Construction_Graph import Hnsw
from itertools import chain
from src.utils import *

"""
File containing the implementation of the DeepGraphConv model, this model was not benchmarked in the paper as we didn't go through with its analysis  but is included because we believe it could be useful for future work
"""


class DeepGraphConv_Surv(torch.nn.Module):
    """
    Implementation of the DeepGraphConv model
    Args:
        edge_agg (str): Type of edge aggregation to use, either 'spatial' or 'latent', default is 'latent'
        resample (float): Dropout rate, default is 0
        ndim (int): Number of input features, default is 1024
        hidden_dim (int): Number of hidden units, default is 256
        linear_dim (int): Number of linear units, default is 256
        pool (bool): Whether to use pooling, default is False
        use_edges (bool): Whether to use edges, default is False
        dropout (float): Dropout rate, default is 0.25
        n_classes (int): Number of classes, default is 4
        radius (int): Radius for the HNSW graph, default is 8
    """
    def __init__(self, edge_agg='latent', resample=0, ndim=1024, hidden_dim=256, 
        linear_dim=256, pool=False, use_edges=False, dropout=0.25, n_classes=4, radius=8):
        super(DeepGraphConv_Surv, self).__init__()
        self.use_edges = use_edges
        self.resample = resample
        self.edge_agg = edge_agg
        self.pool = pool
        self.radius=radius
        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])

        self.conv1 = GINConv(Seq(nn.Linear(ndim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        self.path_attention_head = Att_Net_Gated_Dual(L=hidden_dim, D=hidden_dim, dropout=dropout, n_tasks=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)
    
    def relocate(self):
        from torch_geometric.nn import DataParallel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.conv1 = nn.DataParallel(self.conv1, device_ids=device_ids).to('cuda:0')
            self.conv2 = nn.DataParallel(self.conv2, device_ids=device_ids).to('cuda:0')
            self.conv3 = nn.DataParallel(self.conv3, device_ids=device_ids).to('cuda:0')
            self.path_attention_head = nn.DataParallel(self.path_attention_head, device_ids=device_ids).to('cuda:0')

        self.path_rho = self.path_rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, feat, edge_latent, edge_index=None):
        
        x = feat.squeeze(0)
        if self.edge_agg == 'spatial':
            edge_index = edge_index
        elif self.edge_agg == 'latent':
            edge_index = edge_latent
            
        # batch = data.batch
        edge_attr = None

        if self.resample:
            x = self.fc(x)

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        if self.pool:
            x1, edge_index, _, batch, perm, score = self.pool1(x1, edge_index, None, batch)
            x1_cat = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        if self.pool:
            x2, edge_index, _, batch, perm, score = self.pool2(x2, edge_index, None, batch)
            x2_cat = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))

        h_path = x3

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #Y_prob = F.softmax(logits, dim=1)
        Y_prob = F.sigmoid(logits)
        
        return logits, Y_prob, Y_hat

    def compute_latent_similarity(self, X): 
        model = Hnsw(space='l2')  # Initialize Hnsw with the 'l2' distance metric.
        features = np.array(X.squeeze(0))
        num_patches = features.shape[0]
        
        model.fit(features)  # Fit the model using embeddings.
        a = np.repeat(range(num_patches), self.radius-1)  # Source nodes for the edges.
        b = np.fromiter(chain(*[model.query(features[v_idx-1], topn=self.radius)[1:] for v_idx in range(num_patches)]), dtype=int)
        edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor).to_sparse() # Construct edge tensor.
        print(f'shape edge_latent : {type(edge_latent)}')
        return edge_latent

    def calculate_objective(self, X, Y):
        Y = Y.float()
        edge_latent = self.compute_latent_similarity(X)
        logits, Y_prob, _ = self.forward(X, edge_latent)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = neg_log_bernouilli(Y, Y_prob)
        
        #we don't care about the second argument returned
        return loss, Y_prob

    def calculate_classification_error(self, X, labels, edge_index=None, edge_latent=None):
        _, _, Y_hat = self.forward(X, edge_index, edge_latent)
        error = torch.mean((Y_hat != labels).float())
        return error, Y_hat
        
