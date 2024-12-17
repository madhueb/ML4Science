import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils import *    
import argparse
from pathlib import Path
import json

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
    if model['name'] == 'dsmil': 
        i_classifier = model['i_classifier']( **{key: value for key, value in model['i_classifier_param'].items() 
                                       if key in model['i_classifier'].__init__.__code__.co_varnames})
        b_classifier = model['b_classifier']( **{key: value for key, value in model['b_classifier_param'].items() 
                                       if key in model['b_classifier'].__init__.__code__.co_varnames})
        model_class = model['class'](i_classifier=i_classifier,
                                     b_classifier=b_classifier,
                                   **{key: value for key, value in model['param'].items() 
                                       if key in model['class'].__init__.__code__.co_varnames})

    else : 
        model_class = model['class']( **{key: value for key, value in model['param'].items() 
                                       if key in model['class'].__init__.__code__.co_varnames})

    # cross_validation
    if (json['cross_val'] == True): 
        X_cross_val = np.concatenate((X_train, X_test), axis=0)
        y_cross_val = np.concatenate((y_train, y_test), axis=0)
        cross_val_dataset = TensorDataset(torch.tensor(X_cross_val, dtype=torch.float32),
                                    torch.tensor(y_cross_val, dtype=torch.int))
        cross_param = json['cross_val_param']
        k= cross_param['k']
        epochs = cross_param['epoch']
        k_fold_cross_validation(test_dataset, model_class,  k, epochs)


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
        test(test_loader,y_test,model_class)

        
        

if __name__=="__main__":

    
    parser = argparse.ArgumentParser(description='MIL Pipeline')
    parser.add_argument('--model_config', type=Path, required=True, help='Path to the model configuration file')
    
    args = parser.parse_args()

    with open(args.model_config, 'r') as f:
        model_config = json.load(f)


    pipeline_mil(model_config)


