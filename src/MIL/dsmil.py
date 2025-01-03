import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils import neg_log_bernouilli
import numpy as np


"""
File containing the Deep Supervised Multiple Instance Learning model
"""
class FCLayer(nn.Module):
    """
    Fully connected layer
    Args:
        in_size (int): input size
        out_size (int): output size
    """
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    """
    Instance classifier
    Args:
        feature_size (int): size of the feature vector
        output_class (int): number of classes
        feature_extractor (nn.Module): feature extractor
    """
    def __init__(self, feature_size, output_class, feature_extractor=nn.Identity()):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        x = x.squeeze(0)
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    """
    Bag classifier
    Args:
        input_size (int): input size
        output_class (int): number of classes
        dropout_v (float): dropout value
        nonlinear (bool): whether to use non-linearity
        passing_v (bool): whether to pass the instance representation to the bag classifier
    """

    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class MILNet(nn.Module):
    """
    Deep Supervised Multiple Instance Learning model
    Args:
        i_classifier (nn.Module): instance classifier
        b_classifier (nn.Module): bag classifier
        threshold (float): threshold for classification
    """
    def __init__(self, i_classifier, b_classifier, threshold):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.threshold = threshold
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B

     
    def calculate_objective(self, X, Y):
        Y = Y.float()
        pred_ins, pred_bag, _, _ = self.forward(X)

        #avoid division by zero
        pred_bag = torch.clamp(pred_bag, min=1e-5, max=1. - 1e-5)
        pred_ins = torch.clamp(pred_ins, min=1e-5, max=1. - 1e-5)

        max_pred, _ = torch.max(pred_ins, 0)  

        #from the code, they compute the loss for each classifier score and then take the average
        max_loss = neg_log_bernouilli(Y, torch.sigmoid(max_pred))
        bag_loss = neg_log_bernouilli(Y, torch.sigmoid(pred_bag))
        loss = 0.5*bag_loss + 0.5*max_loss
      
        #we don't care about the second argument returned
        return loss, pred_bag

    def calculate_classification_error(self, X, labels):
        pred_ins,pred_bag,_, _= self.forward(X)
        pred_ins, _ = torch.max(pred_ins, 0)  
        prediction = (1/2) * torch.sigmoid(pred_ins) + torch.sigmoid(pred_bag)
        
        class_pred = (prediction >= self.threshold).int()
        error = torch.mean((class_pred!= labels).float())
        return error, class_pred, prediction
