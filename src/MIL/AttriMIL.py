import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from src.MIL.ABMIL import get_attn_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

    

class AttriMIL(nn.Module): 
    '''
    Multi-Branch ABMIL with constraints
    '''
    def __init__(self, n_classes=2, embed_size=1024):
        super().__init__()
        self.adaptor = nn.Sequential(nn.Linear(embed_size, embed_size//2),
                                     nn.ReLU(),
                                     nn.Linear(embed_size // 2 , embed_size))
        
        attention = []
        classifer = [nn.Linear(embed_size, 1) for i in range(n_classes)]
        for i in range(n_classes):
            attention.append(get_attn_module(embed_size=embed_size, hidden_size=embed_size//2, att_branches=1))
        self.attention_nets = nn.ModuleList(attention)
        self.classifiers = nn.ModuleList(classifer)
        self.n_classes = n_classes
        self.bias = nn.Parameter(torch.zeros(n_classes), requires_grad=True)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, h):
        h = h.squeeze(0)
        h = h + self.adaptor(h)
        A_raw = torch.empty(self.n_classes, h.size(0), ) # N x 1
        instance_score = torch.empty(1, self.n_classes, h.size(0)).float().to(h.device)
        for c in range(self.n_classes):
            A, h = self.attention_nets[c](h)
            A = torch.transpose(A, 1, 0)  # 1 x N
            A_raw[c] = A
            instance_score[0, c] = self.classifiers[c](h)[:, 0]
        attribute_score = torch.empty(1, self.n_classes, h.size(0)).float().to(h.device)
        for c in range(self.n_classes):
            attribute_score[0, c] = instance_score[0, c] * torch.exp(A_raw[c])
            
        logits = torch.empty(1, self.n_classes).float().to(h.device)
        for c in range(self.n_classes):
            logits[0, c] = torch.sum(attribute_score[0, c], keepdim=True, dim=-1) / torch.sum(torch.exp(A_raw[c]), dim=-1) + self.bias[c]
            
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {}
        return logits, Y_prob, Y_hat, attribute_score, results_dict
    
    
    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
            Y = Y.float()
            _,_, Y_hat, _,_ = self.forward(X)
            error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

            return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.long()
        Y = Y.unsqueeze(0)  
        logits, Y_prob, _, A, _ = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        loss = self.loss_fn(logits, Y)

        return loss, A

