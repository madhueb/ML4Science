import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from src.MIL.ABMIL import get_attn_module
from src.utils import neg_log_bernouilli

"""
File containing the implementation of the VarMIL model
"""

class VarMIL( nn.Module):
    """
    Implementation of the VarMIL model
    Args: 
        embed_size (int): Dimension of the encoder features, default is 1024
        hidden_size (int): Dimension of the hidden layer, default is 128
        gated (bool): If True, use the gated attention mechanism, default is True
        separate_attn (bool): If True, use separate attention mechanisms for mean and variance, default is False
        n_var_pools (int): Number of variance pooling projections, default is 100
        act_func (str): The activation function to apply to variance pooling, default is 'sqrt'
        log_eps (float): Epsilon value for log(epsilon + ), default is 0.01
        dropout (float): Dropout rate, default is 0.0
    """
    def __init__(self, embed_size=1024, 
                 hidden_size=128, gated=True,
                 separate_attn=False, n_var_pools=100,
                 act_func='sqrt', log_eps=0.01,
                 dropout=0.):
        super().__init__()


        self.separate_attn = bool(separate_attn)

        if self.separate_attn:
            attention = get_attn_module(embed_size=embed_size,
                                        hidden_size=hidden_size,
                                        att_branches =2,
                                        dropout=dropout,
                                        gated=gated)


        else:
            attention = get_attn_module(embed_size=embed_size,
                                        hidden_size=hidden_size,
                                        att_branches =1,
                                        dropout=dropout,
                                        gated=gated)

            
        self.enc_and_attend = attention

        # variance pooling
        self.var_pool = VarPool(embed_size=embed_size,
                                n_var_pools=n_var_pools,
                                log_eps=log_eps,
                                act_func=act_func)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_size + n_var_pools, 1),
            nn.Sigmoid()
        )

    def get_encode_and_attend(self, bag):

        # Get attention scores 
        attn_scores,_ = self.enc_and_attend.forward(bag.squeeze(0))
        attn_scores =torch.transpose(attn_scores, 1, 0)
        attn_scores = F.softmax(attn_scores, dim=1)


        if self.separate_attn:  
            mean_attn = attn_scores[0]

            var_attn = attn_scores[1]

        else:
            
            mean_attn = attn_scores
            var_attn = attn_scores

        return bag, mean_attn, var_attn

    def forward(self, bag):

        bag_feats, mean_attn, var_attn = self.get_encode_and_attend(bag)

        # (batch_size, n_instances, encode_dim) -> (batch_size, embed_size)
        mean_attn = mean_attn.unsqueeze(-1)
        weighted_avg_bag_feats = (bag_feats * mean_attn).sum(1)

        var_pooled_bag_feats = self.var_pool(bag_feats, var_attn)

        ################################
        # get output from head network #
        ################################
        merged_bag_feats = \
            torch.cat((weighted_avg_bag_feats, var_pooled_bag_feats),
                      dim=1)
        # (batch_size, encode_dim +  n_var_pool)
        

        Y_prob = self.classifier(merged_bag_feats)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, merged_bag_feats
        
    def calculate_classification_error(self, X, Y):
            Y = Y.float()
            Y_prob, Y_hat, _ = self.forward(X)
            error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

            return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = neg_log_bernouilli(Y, Y_prob)
        return loss, A



class VarPool(nn.Module):
    """
    Variance pooling layer
    Args:
        embed_size (int): Dimension of the encoder features
        n_var_pools (int): Number of variance pooling projections
        act_func (str): The activation function to apply to variance pooling, between 'sqrt', 'log', 'sigmoid', 'identity', default is 'sqrt'
        log_eps (float): Epsilon value for log, default is 0.01
    """
    def __init__(self, embed_size, n_var_pools, act_func='sqrt', log_eps=0.01):
        super().__init__()
        assert act_func in ['sqrt', 'log', 'sigmoid', 'identity']

        self.var_projections = nn.Linear(embed_size, int(n_var_pools),
                                         bias=False)
        self.act_func = act_func
        self.log_eps = log_eps

    def init_var_projections(self):
        """
        Initializes the variance projections from isotropic gaussians such that each projections expected norm is 1
        """
        embed_size, n_pools = self.var_projections.weight.data.shape

        self.var_projections.weight.data = \
            torch.normal(mean=torch.zeros(embed_size, n_pools),
                         std=1/np.sqrt(embed_size))


    def get_proj_attn_weighted_resids_sq(self, bag, attn, return_resids=False):
        """
        Computes the attention weighted squared residuals of each instance to the projection mean.

        Args:
            bag: (batch_size, n_instances, instance_dim)
                The bag features.

            attn: (batch_size, n_instances, 1)
                The normalized instance attention scores.

            return_resids: bool
                If True, return the residuals as well.
        """
        assert len(bag.shape) == 3, \
            "Be sure to include batch in first dimension"

        projs = self.var_projections(bag)
        # (batch_size, n_instances, n_var_pools)

        if attn is None:
            attn = 1 / projs.shape[1]

        attn = attn.unsqueeze(-1)
        proj_weighted_avg = (projs * attn).sum(1)
        # (batch_size, n_var_pools)

        resids = projs - proj_weighted_avg.unsqueeze(1)
        attn_resids_sq = attn * (resids ** 2)
        # (batch_size, n_instances, n_var_pools)

        if return_resids:
            return attn_resids_sq, resids
        else:
            return attn_resids_sq

    def forward(self, bag, attn):

        attn_resids_sq = self.\
            get_proj_attn_weighted_resids_sq(bag, attn, return_resids=False)
        # (batch_size, n_instances, n_var_pools)

        var_pool = (attn_resids_sq).sum(1)
        # (batch_size, n_var_pools)

        if self.act_func == 'sqrt':
            return torch.sqrt(var_pool)

        elif self.act_func == 'log':
            return torch.log(self.log_eps + var_pool)

        elif self.act_func == 'sigmoid':
            return torch.sigmoid(var_pool)

        elif self.act_func == 'identity':
            return var_pool