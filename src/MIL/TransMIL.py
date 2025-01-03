import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
import torch.nn.functional as F
from src.utils import neg_log_bernouilli


""" 
File containing the implementation of the TransMIL model
"""
class TransLayer(nn.Module):
    """
    Implementation of the Transformer layer
    Args:
        norm_layer (nn.Module): Normalization layer to use, default is nn.LayerNorm
        dim (int): Dimension of the input, default is 512
    """

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    """
    Implementation of the Positional Encoding with Positional Encoding Gradients (PPEG) layer
    Args:
        dim (int): Dimension of the input, default is 512
    """
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    """
    Implementation of the TransMIL model
    Args:
        n_classes (int): Number of classes, default is 2
        embed_size (int): Size of the embeddings, default is 1024
    """
    def __init__(self, n_classes,embed_size=1024):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(embed_size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, X):
        h = X.float()  # [B, n, 1024]
        h = self._fc1(h)  # [B, n, 512]
        
        # Pad to nearest square
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # Add cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # TransLayer x1
        h = self.layer1(h)  # [B, N, 512]

        # PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # TransLayer x2
        h = self.layer2(h)  # [B, N, 512]

        # Extract cls_token
        h = self.norm(h)[:, 0]

        # Predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)
        return logits, Y_prob, Y_hat



    def calculate_objective(self, X, Y):
        Y = Y.float()
        logits, Y_prob, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = neg_log_bernouilli(Y, Y_prob[0,1])
        
        #we don't care about the second argument returned
        return loss, Y_prob

    def calculate_classification_error(self, X, labels):
        _, Y_prob, Y_hat = self.forward(X)
        error = torch.mean((Y_hat != labels).float())
        return error, Y_hat, [Y_prob[0,1]]


    