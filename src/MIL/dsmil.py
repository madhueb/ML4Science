import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import neg_log_bernouilli
import numpy as np

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
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
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B

     
    def calculate_objective(self, X, Y):
        Y = Y.float()
        pred_ins, pred_bag, _, _ = self.forward(X)

        pred_bag = torch.clamp(pred_bag, min=1e-5, max=1. - 1e-5)
        pred_ins = torch.clamp(pred_ins, min=1e-5, max=1. - 1e-5)

        pred_ins, _ = torch.max(pred_ins, 0)  
        
        #from the code, they compute the loss for each classifier score and then take the average
        ins_loss = neg_log_bernouilli(Y, pred_ins)
        bag_loss = neg_log_bernouilli(Y, pred_bag)
        total_loss = 0.5*bag_loss + 0.5*ins_loss
      
        #we don't care about the second argument returned
        return total_loss, pred_bag

    def calculate_classification_error(self, X, labels):
        pred_ins,pred_bag,_, _= self.forward(X)
        pred_ins, _ = torch.max(pred_ins, 0)  
        prediction = (1/2) * (torch.sigmoid(pred_ins) + torch.sigmoid(pred_bag))
        #need to fine tune threshold
        threshold = 0.5
        class_pred = (prediction >= threshold).int()
        error = torch.mean((class_pred!= labels).float())
        return error, class_pred

#-------------------------------------------------

    def multi_label_roc(labels, predictions, num_classes, pos_label=1):
        fprs = []
        tprs = []
        thresholds = []
        thresholds_optimal = []
        aucs = []
        if len(predictions.shape)==1:
            predictions = predictions[:, None]
        print(f"labels shape in multi label roc : {labels.shape}")
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=-1)
        for c in range(0, num_classes):
            label = labels[:, c]
            prediction = predictions[:, c]
            fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
            fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
            # c_auc = roc_auc_score(label, prediction)
            try:
                c_auc = roc_auc_score(label, prediction)
                print("ROC AUC score:", c_auc)
            except ValueError as e:
                if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                    print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                    c_auc = 1
                else:
                    raise e

            aucs.append(c_auc)
            thresholds.append(threshold)
            thresholds_optimal.append(threshold_optimal)
        return aucs, thresholds, thresholds_optimal

    def optimal_thresh(fpr, tpr, thresholds, p=0):
        loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
        idx = np.argmin(loss, axis=0)
        return fpr[idx], tpr[idx], thresholds[idx]
"""
test part that determinies the final score and final label given the optimal threshold 
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    class_prediction_bag = copy.deepcopy(test_predictions)
    class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
    class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
    test_predictions = class_prediction_bag
    test_labels = np.squeeze(test_labels)
   
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    if return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal
"""
    