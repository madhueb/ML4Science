import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils import *    
import argparse
from pathlib import Path
import json
import src
from src.MIL.ABMIL import *
from src.MIL.VarMIL import *
from src.MIL.CLAM import *
from src.MIL.TransMIL import *
import src.MIL.dsmil as dsmil
from src.MIL.ACMIL import *
from src.MIL.AttriMIL import *
#import src.MIL.DeepGraphConv as dgc
from src.utils import *
import pandas as pd

def get_class(model_name):
    if model_name == 'dsmil':
        return dsmil.MILNet
    elif model_name == 'Emb_max':
        return Emb_max
    elif model_name == 'Emb_mean':
        return Emb_mean
    elif model_name == 'Attention':
        return Attention
    elif model_name == 'GatedAttention':
        return GatedAttention
    elif model_name == 'VarMIL':
        return VarMIL
    elif model_name == 'CLAM':
        return CLAM_SB
    elif model_name == 'TransMIL':
        return TransMIL
    elif model_name == 'ACMIL':
        return ACMIL_GA
    elif model_name == 'AttriMIL':
        return AttriMIL
    else:
        raise ValueError(f'Unknown model name: {model_name}')

def pipeline_mil(json): 

    #load data 
    train_dict = load_data(json['train_file_path'])
    test_dict = load_data(json['test_file_path'])

    X_train = train_dict['embeddings'][:,1:,:]
    y_train = train_dict['labels']
    
    X_test = test_dict['embeddings'][:,1:,:]
    y_test = test_dict['labels']

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                               torch.tensor(y_train, dtype=torch.int))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                              torch.tensor(y_test, dtype=torch.int))


    #create model 
    model = json['model']
    model_class = get_class(model['name'])
    if model['name'] == 'dsmil': 
        i_classifier = dsmil.IClassifier( **{key: value for key, value in model['i_classifier_param'].items() 
                                       if key in dsmil.IClassifier.__init__.__code__.co_varnames})
        b_classifier = dsmil.BClassifier( **{key: value for key, value in model['b_classifier_param'].items() 
                                       if key in dsmil.BClassifier.__init__.__code__.co_varnames})
        model_class = model_class(i_classifier=i_classifier,
                                     b_classifier=b_classifier,
                                   **{key: value for key, value in model['param'].items() 
                                       if key in model_class.__init__.__code__.co_varnames})

    else : 
        model_class = model_class( **{key: value for key, value in model['param'].items() 
                                       if key in model_class.__init__.__code__.co_varnames})

    # cross_validation
    if (json['cross_val'] == True): 
        X_cross_val = np.concatenate((X_train, X_test), axis=0)
        y_cross_val = np.concatenate((y_train, y_test), axis=0)
        cross_val_dataset = TensorDataset(torch.tensor(X_cross_val, dtype=torch.float32),
                                    torch.tensor(y_cross_val, dtype=torch.int))
        cross_param = json['cross_val_param']
        k= cross_param['k']
        epochs = cross_param['epoch']
        k_fold_cross_validation(cross_val_dataset, model_class,  k, epochs)


    else : 
        train_param = json['training']
        epoch = train_param['epoch']
        lr = train_param['lr']
        weight_decay=train_param['weight_decay']
        batch_size = train_param['batch_size']

        print('----------Start Training----------')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train(train_loader,epoch,model_class,lr, weight_decay,print_results=False)
        
        print('----------Start Testing----------')
        _, _, _, _,_, fpr, tpr = test(test_loader,y_test,model_class)

        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        name = model['name']
        df.to_csv(f'{name}.csv',index=False)

        
        

if __name__=="__main__":

    
    parser = argparse.ArgumentParser(description='MIL Pipeline')
    parser.add_argument('--model_config', type=Path, required=True, help='Path to the model configuration file')
    
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_config = json.load(f)


    pipeline_mil(model_config)


