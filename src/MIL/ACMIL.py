import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from MIL_layers import get_attn_module

class ACMIL(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes=2):
        """
        ACMIL Model: Uses attribute-guided attention for bag-level prediction.
        Args:
            input_dim (int): Dimension of instance-level input features.
            latent_dim (int): Dimension of latent space for attention.
            num_classes (int): Number of output classes (default=2).
        """
        super(ACMIL, self).__init__()
        self.attention = get_attn_module(input_dim,latent_dim, att_branches=1)
        self.classifier = nn.Linear(input_dim, num_classes)  # Classifier layer

    def forward(self, x):
        """
        Forward pass of the ACMIL model.
        Args:
            x (Tensor): Input bag features of shape (N, D), where
                        N = number of instances in the bag
                        D = feature dimension.
        Returns:
            bag_logits (Tensor): Predicted logits for the bag.
            attention_weights (Tensor): Attention weights for each instance.
        """
        attention_weights = self.attention(x)  # Compute attention weights
        attention_weights = F.softmax(attention_weights, dim=0)  # Normalize scores
        bag_representation = torch.sum(attention_weights * x, dim=0)  # Weighted sum
        bag_logits = self.classifier(bag_representation)  # Classify bag

        #add Madeleine
        Y_prob = F.softmax(bag_logits, dim = 1)
        Y_hat = torch.topk(bag_logits, 1, dim = 1)[1]
        return bag_logits, Y_prob, Y_hat, attention_weights


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
