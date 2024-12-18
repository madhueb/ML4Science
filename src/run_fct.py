import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils import * 
from sklearn.model_selection import train_test_split   

from src.MIL.ABMIL import *
from src.MIL.VarMIL import *
from src.MIL.CLAM import *
from src.MIL.TransMIL import *
import src.MIL.dsmil as dsmil
from src.MIL.ACMIL import *
from src.MIL.AttriMIL import *
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


# FUNCTIONS TO RETRIEVE DATASETS #########################################
def load_data(file_path): 
    """
    Load data for TCGA dataset 
    """
    with open(file_path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_mil_data(dataset_name):

    if dataset_name not in ['Fox','Elephant','Tiger','musk1','musk2']:
        raise ValueError("Name of the dataset not recognized")
    if dataset_name =='Elephant' or dataset_name == 'Fox' or dataset_name == 'Tiger' :
        file_path = 'datasets/mil_dataset/'+dataset_name +'/data_100x100.svm'
    else :
        file_path = 'datasets/mil_dataset/Musk/'+dataset_name +'norm.svm'

    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = df[df.columns[0]]

    
    data_list = []
    for i in range(0, df.shape[0]):  
        data = str(df.iloc[i]).split(' ')
        ids = data[0].split(':')
        idi = int(ids[0]) #id of the instance
        idb = int(ids[1]) #id of the bag
        idc = int(ids[2]) #id of the class
        data = data[1:]
        feature_vector = np.zeros(len(data))  
        for i, feature in enumerate(data):
            feature_data = feature.split(':')
            if len(feature_data) == 2:
                feature_vector[i] = feature_data[1]
        data_list.append([idi, idb, idc, feature_vector])

    data =[]
    num_bags = len(set([x[1] for x in data_list]))
    for i in range(num_bags):
        class_bag = [x[2] for x in data_list if x[1] == i][0]
        class_bag = 1 if class_bag == 1 else 0
        embedding = torch.tensor([x[3] for x in data_list if x[1] == i], dtype=torch.float32)
        data.append((embedding,torch.tensor(class_bag)))

    return data

# FUNCTION TO RUN #########################################

def run_TCGA(json):
    #load data 
    train_dict = load_data('datasets/TCGA/train_dict.pkl')
    test_dict = load_data('datasets/TCGA/test_dict.pkl')

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
        df.to_csv(f'csv_files/TCGA/{name}.csv',index=False)

def run_mil_dataset(json,dataset): 

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

#def run_c16(json):

