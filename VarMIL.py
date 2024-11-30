"""
Variance Pooling + Attention based multiple instance learning architecture.
"""
import torch
import torch.nn as nn

from numbers import Number
import numpy as np


from MIL_layers import get_attn_module



class AttnMeanAndVarPoolMIL( nn.Module):
    """
    Attention mean  and variance pooling architecture.

    Parameters
    ----------
    encoder_dim: int
        Dimension of the encoder features. This is either the dimension output by the instance encoder (if there is one) or it is the dimension of the input feature (if there is no encoder).

    encoder: None, nn.Module
        (Optional) The bag instance encoding network.

    head: nn.Module, int, tuple of ints
        (Optional) The network after the attention mean pooling step. If an int is provided a single linear layer is added. If a tuple of ints is provided then a multi-layer perceptron with RELU activations is added.

    n_attn_latent: int, None
        Number of latent dimension for the attention layer. If None, will default to (n_in + 1) // 2.

    gated: bool
        Use the gated attention mechanism.

    separate_attn: bool
        WHether or not we want to use separate attention branches for the mean and variance pooling.

    n_var_pools: int
        Number of variance pooling projections.

    act_func: str
        The activation function to apply to variance pooling. Must be one of ['sqrt', 'log', 'sigmoid'].

    log_eps: float
        Epsilon value for log(epsilon + ) var pool activation function.

    dropout: bool, float
        Whether or not to use dropout in the attention mechanism. If True, will default to p=0.25.

    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """
    def __init__(self, encoder_dim, 
                 n_attn_latent=None, gated=True,
                 separate_attn=False, n_var_pools=100,
                 act_func='sqrt', log_eps=0.01,
                 dropout=False):
        super().__init__()

        ###########################
        # Setup encode and attend #
        ###########################
        self.separate_attn = bool(separate_attn)

        if self.separate_attn:
            attention = get_attn_module(encoder_dim=encoder_dim,
                                        n_attn_latent=n_attn_latent,
                                        att_branches =2,
                                        dropout=dropout,
                                        gated=gated)


        else:
            attention = get_attn_module(encoder_dim=encoder_dim,
                                        n_attn_latent=n_attn_latent,
                                        att_branches =1,
                                        dropout=dropout,
                                        gated=gated)

            
        self.enc_and_attend = attention

        ####################
        # Variance pooling #
        ####################
        self.var_pool = VarPool(encoder_dim=encoder_dim,
                                n_var_pools=n_var_pools,
                                log_eps=log_eps,
                                act_func=act_func)
        
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim + n_var_pools, 1),
            nn.Sigmoid()
        )

    def get_encode_and_attend(self, bag):

        ###################################
        # Instance encoding and attention #
        ###################################

        # instance encodings and attention scores

        _,_,attn_scores = self.enc_and_attend.forward(bag)


        # normalize attetion
        if self.separate_attn:  
            mean_attn = attn_scores[0]

            var_attn = attn_scores[1]

        else:
            
            mean_attn = attn_scores
            var_attn = attn_scores

        return bag, mean_attn, var_attn

    def forward(self, bag):

        bag_feats, mean_attn, var_attn = self.get_encode_and_attend(bag)

        #####################
        # Attention pooling #
        #####################

        # (batch_size, n_instances, encode_dim) -> (batch_size, encoder_dim)
        mean_attn = mean_attn.unsqueeze(-1)
        weighted_avg_bag_feats = (bag_feats * mean_attn).sum(1)

        var_pooled_bag_feats = self.var_pool(bag_feats, var_attn)
        # (batch_size, n_var_pool)

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
            _, Y_hat, _ = self.forward(X)
            error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

            return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A



class VarPool(nn.Module):
    """
    A variance pooling layer.

    Compute the variance across attended & projected instances

    Parameters
    ----------
    encoder_dim: int
        Dimension of the encoder features.

    n_var_pools: int
        Number of variance pooling projections.

    act_func: str
        The activation function to apply to variance pooling. Must be on of ['sqrt', 'log', 'sigmoid', 'identity'].

    log_eps: float
        Epsilon value for log(epsilon + ).

    apply_attn: bool
        If True, apply attn to var projection. If False, do not apply attn (Mainly for SumMIL)

    """
    def __init__(self, encoder_dim, n_var_pools, act_func='sqrt', log_eps=0.01):
        super().__init__()
        assert act_func in ['sqrt', 'log', 'sigmoid', 'identity']

        self.var_projections = nn.Linear(encoder_dim, int(n_var_pools),
                                         bias=False)
        self.act_func = act_func
        self.log_eps = log_eps

    def init_var_projections(self):
        """
        Initializes the variance projections from isotropic gaussians such that each projections expected norm is 1
        """
        encoder_dim, n_pools = self.var_projections.weight.data.shape

        self.var_projections.weight.data = \
            torch.normal(mean=torch.zeros(encoder_dim, n_pools),
                         std=1/np.sqrt(encoder_dim))

    # def get_projection_vector(self, idx):
    #     """
    #     Returns a projection vector
    #     """
    #     return self.var_projections.weight.data[idx, :].detach()

    def get_proj_attn_weighted_resids_sq(self, bag, attn, return_resids=False):
        """
        Computes the attention weighted squared residuals of each instance to the projection mean.

        Parameters
        ----------
        bag: (batch_size, n_instances, instance_dim)
            The bag features.

        attn: (batch_size, n_instances, 1)
            The normalized instance attention scores.

        Output
        ------
        attn_resids_sq: (batch_size, n_instances, n_var_pools)
            The attention weighted squared residuals.

        if return_resids is True then we also return resids

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
        """
        Parameters
        ----------
        bag: (batch_size, n_instances, instance_dim)
            The bag features.

        attn: (batch_size, n_instances, 1)
            The normalized instance attention scores.

        Output
        ------
        var_pool: (batch_size, n_var_pools)
        """
        attn_resids_sq = self.\
            get_proj_attn_weighted_resids_sq(bag, attn, return_resids=False)
        # (batch_size, n_instances, n_var_pools)

        # computed weighted average -- note this effectively uses
        # denominator 1/n since the attn sum to one.
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