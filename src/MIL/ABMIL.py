import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import *

"""
File containing the implementation of the Attention Based MIL models and some baselines (Emb_mean, Emb_max)
"""

class Att_net(nn.Module):
    """
    Attention network
    Args:
        embed_size (int): Size of the input embeddings, default is 1024
        hidden_size (int): Size of the hidden layer, default is 512
        ATTENTION_BRANCHES (int): Number of attention branches, default is 1
        dropout (float): Dropout rate, default is 0.
    """
    def __init__(self, embed_size=1024, hidden_size=512, ATTENTION_BRANCHES=1,dropout=0.):
        super(Att_net, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES

        self.attention = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.ATTENTION_BRANCHES)
        )

    def forward(self, x):
        
        A = self.attention(x)  # KxATTENTION_BRANCHES
        return A,x

class Att_Net_Gated_Dual(nn.Module):
    """
    Attention network with gated mechanism for dual attention
    Args:
        L (int): Size of the input embeddings, default is 1024
        D (int): Size of the hidden layer, default is 256
        dropout (bool): Whether to use dropout, default is False
        n_tasks (int): Number of tasks, default is 1
    """

    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Att_Net_Gated_Dual, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class GatedAtt_net(nn.Module):
    """
    Gated Attention network
    Args:
        embed_size (int): Size of the input embeddings, default is 1024
        hidden_size (int): Size of the hidden layer, default is 128
        ATTENTION_BRANCHES (int): Number of attention branches, default is 1
        dropout (float): Dropout rate, default is 0.
    """
    def __init__(self, embed_size=1024, hidden_size=128, ATTENTION_BRANCHES=1,dropout=0.):
        super(GatedAtt_net, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES

        self.attention_U = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.attention_w = nn.Linear(self.hidden_size, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

    def forward(self, x):
        A_V = self.attention_V(x)  # KxL
        A_U = self.attention_U(x)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        return A,x

class Attention(nn.Module):
    """
    Attention layer
    Args:
        embed_size (int): Size of the input embeddings, default is 1024
        hidden_size (int): Size of the hidden layer, default is 128
        ATTENTION_BRANCHES (int): Number of attention branches, default is 1
        dropout (float): Dropout rate, default is 0.
    """
    def __init__(self, embed_size=1024, hidden_size=128, ATTENTION_BRANCHES=1,dropout=0.):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES

        self.attention = Att_net(embed_size, hidden_size, ATTENTION_BRANCHES,dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        A,_ = self.attention(x)  # KxATTENTION_BRANCHES
        
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
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

class GatedAttention(nn.Module):
    """
    Gate Attention layer
    Args:
        embed_size (int): Size of the input embeddings, default is 1024
        hidden_size (int): Size of the hidden layer, default is 128
        ATTENTION_BRANCHES (int): Number of attention branches, default is 1
        dropout (float): Dropout rate, default is 0."""
    def __init__(self, embed_size=1024, hidden_size=128, ATTENTION_BRANCHES=1,dropout=0.):
        super(GatedAttention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES


        self.attention = GatedAtt_net(embed_size, hidden_size, ATTENTION_BRANCHES,dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        A,_ = self.attention(x)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = neg_log_bernouilli(Y, Y_prob)
        return loss, A
    
class Emb_mean(nn.Module):  
    """
    Embedding mean layer
    Args:
        embed_size (int): Size of the input embeddings, default is 1024
    """
    def __init__(self, embed_size=1024): 
        super(Emb_mean, self).__init__()  
        self.embed_size = embed_size

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        Z = torch.mean(x,0).unsqueeze(0)
        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat
    
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat, Y_prob
    
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = neg_log_bernouilli(Y, Y_prob)
        return loss, None

class Emb_max(nn.Module): 
    """
    Embedding max layer
    Args:
        embed_size (int): Size of the input embeddings, default is 1024
    """ 
    def __init__(self, embed_size=1024): 
        super(Emb_max, self).__init__() 
        self.embed_size = embed_size

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        Z = torch.max(x,0)[0].unsqueeze(0)
        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat
    
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat, Y_prob
    
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = neg_log_bernouilli(Y, Y_prob)
        return loss, None


def get_attn_module(embed_size, hidden_size, att_branches, dropout=0., gated=False):
    """
    Gets the attention module for other models
    """
    if gated:
        return GatedAtt_net(embed_size = embed_size, hidden_size = hidden_size,ATTENTION_BRANCHES=att_branches,dropout=dropout)
    else:
        return Att_net(embed_size=embed_size, hidden_size =hidden_size,ATTENTION_BRANCHES=att_branches,dropout=dropout)
