import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.MIL.ABMIL import get_attn_module

"""
File containing the implementation of the ACMIL model
"""

class Classifier_1fc(nn.Module):
    """
    Classifier with one fully connected layer
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of classes
        droprate (float): Dropout rate, default is 0.
    """
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    
class residual_block(nn.Module):
    """
    Residual block
    Args:
        nChn (int): Number of input channels, default is 512
    """
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x
    
class DimReduction(nn.Module):
    """
    Dimensionality reduction module
    Args:
        n_channels (int): Number of input channels
        m_dim (int): Dimension of the output, default is 512
        numLayer_Res (int): Number of residual blocks, default is 0
    """
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x
    
class ACMIL_MHA(nn.Module):
    """
    ACMIL model with Multi-Head Attention
    Args:
        embed_size (int): Size of the input embeddings
        hidden_size (int): Size of the hidden layer
        n_classes (int): Number of classes
        n_token (int): Number of attention tokens, default is 1
        n_masked_patch (int): Number of masked patches, default is 0
        mask_drop (float): Mask drop rate, default is 0
    """
    def __init__(self, embed_size,hidden_size,n_classes, n_token=1, n_masked_patch=0, mask_drop=0):
        super(ACMIL_MHA, self).__init__()
        self.dimreduction = DimReduction(embed_size, hidden_size)
        self.sub_attention = nn.ModuleList()
        for i in range(n_token):
            self.sub_attention.append(MutiHeadAttention(hidden_size, 8, n_masked_patch=n_masked_patch, mask_drop=mask_drop))
        self.bag_attention = MutiHeadAttention_modify(hidden_size, 8)
        self.q = nn.Parameter(torch.zeros((1, n_token, hidden_size)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = n_classes

        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(hidden_size, n_classes, 0.0))
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(hidden_size, n_classes, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)

        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))

        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns



class MutiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    Args:
        embedding_dim (int): Size of the input embeddings
        num_heads (int): Number of attention heads
        downsample_rate (int): Downsample rate, default is 1
        dropout (float): Dropout rate, default is 0.1
        n_masked_patch (int): Number of masked patches, default is 0
        mask_drop (float): Mask drop rate, default is 0
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and self.training:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b*h*q, c).to(attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]

class MutiHeadAttention_modify(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    Args:
        embedding_dim (int): Size of the input embeddings
        num_heads (int): Number of attention heads
        downsample_rate (int): Downsample rate, default is 1
        dropout (float): Dropout rate, default is 0.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, v: Tensor, attn: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(v)

        # Separate into heads
        v = self._separate_heads(v, self.num_heads)

        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0]


class ACMIL_GA(nn.Module):
    """
    ACMIL model with Gated Attention
    Args:
        embed_size (int): Size of the input embeddings
        hidden_size (int): Size of the hidden layer
        n_classes (int): Number of classes
        D (int): Dimension of the attention output, default is 128
        droprate (float): Dropout rate, default is 0
        n_token (int): Number of attention tokens, default is 1
        n_masked_patch (int): Number of masked patches, default is 0
        mask_drop (float): Mask drop rate, default is 0
        loss_fn (nn.Module): Loss function, default is nn.CrossEntropyLoss()
    """
    def __init__(self, embed_size,hidden_size,n_classes, D=128, droprate=0, n_token=1, n_masked_patch=0, mask_drop=0,loss_fn = nn.CrossEntropyLoss()):
        super(ACMIL_GA, self).__init__()
        
        self.loss_fn = loss_fn
        self.dimreduction = DimReduction(embed_size, hidden_size)
        
        self.attention = get_attn_module(hidden_size, D, att_branches=n_token,gated=True)
       
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(hidden_size, n_classes, droprate))
        
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop

        self.n_token = n_token
        
        self.Slide_classifier = Classifier_1fc(hidden_size, n_classes, droprate)
        
    def forward(self, x): ## x: N x L
        x = x.squeeze(0)
        x = self.dimreduction(x)
        A,_ = self.attention(x)  ## K x N
        A = torch.transpose(A, 1, 0)



        if self.n_masked_patch > 0 and self.training:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x) ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        Y_i = torch.stack(outputs, dim=0)
        logits = self.Slide_classifier(bag_feat)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits,Y_i, Y_prob, Y_hat, A

    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat
    

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _,_,Y_prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, [Y_prob[0,1]]

    def calculate_objective(self, X, Y):
        Y = Y.long()
        Y = Y.unsqueeze(0)
        logits,Y_i, _,_, A = self.forward(X)

        if self.n_token > 1:
            loss0 = self.loss_fn(Y_i,Y.repeat_interleave(self.n_token))
        else:
            loss0 = torch.tensor(0.)

        loss1 = self.loss_fn(logits, Y)
        diff_loss = torch.tensor(0).to(Y.device, dtype=torch.float)
        A = torch.softmax(A, dim=-1)

        for i in range(self.n_token):
            for j in range(i + 1, self.n_token):
                diff_loss += torch.cosine_similarity(A[:, i], A[:, j], dim=-1).mean() / (
                            self.n_token * (self.n_token - 1) / 2)
        
        loss =loss0 +loss1 + diff_loss
        return loss, A


