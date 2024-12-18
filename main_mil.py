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
from sklearn.model_selection import train_test_split

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

def pipeline_mil(json,dataset): 

    data = get_mil_data(dataset)
    X = [x[0] for x in data]
    y = [x[1] for x in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset =[(X_train[i],y_train[i]) for i in range(len(X_train))]
    test_dataset =[(X_test[i],y_test[i]) for i in range(len(X_test))]

    #Change the embedding size of the model if needed
    json['model']['param']['embed_size'] = X_train[0].shape[1]

    json['cross_val']=False

    if json['model']['name'] == 'dsmil':
        json['model']['i_classifier_param']['feature_size'] = X_train[0].shape[1]
        json['model']['b_classifier_param']['input_size'] = X_train[0].shape[1]

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
        cross_val_dataset = data
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
        df.to_csv(f'csv_files/{dataset}/{name}.csv',index=False)

        
        

if __name__=="__main__":

    
    # parser = argparse.ArgumentParser(description='MIL Pipeline')
    # parser.add_argument('--model_config', type=Path, required=True, help='Path to the model configuration file')
    
    # args = parser.parse_args()

    # with open(args.model_config, 'r') as f:
    for model in ['Emb_mean','Emb_max','Attention','GatedAttention','ACMIL','AttriMIL','CLAM','dsmil']:
        with open('/Users/madeleine/ML4Science/json_files/'+model+'.json', 'r') as f:
            model_config = json.load(f)

        for dataset in ['Fox','Elephant','Tiger','musk1','musk2']:
            print("Results for ",dataset, " with model ",model)
            pipeline_mil(model_config,dataset)



