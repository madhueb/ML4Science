import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split   

from src.MIL.ABMIL import *
from src.MIL.VarMIL import *
from src.MIL.CLAM import *
from src.MIL.TransMIL import *
import src.MIL.dsmil as dsmil
from src.MIL.ACMIL import *
from src.MIL.AttriMIL import *
from src.utils import *



def get_class(model_name):
    """
    Function to get the class of the model
    Args:
        model_name (str): Name of the model
    Returns:
        class (class): Class of the model
    """
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
    """
    Load data for MIL datasets
    Args:
        dataset_name (str): Name of the dataset
    Returns:
        data (list): List of tuples (embedding, label)
    """

    #Get the path to the dataset
    if dataset_name not in ['Fox','Elephant','Tiger','musk1','musk2']:
        raise ValueError("Name of the dataset not recognized")
    if dataset_name =='Elephant' or dataset_name == 'Fox' or dataset_name == 'Tiger' :
        file_path = 'datasets/mil_dataset/'+dataset_name +'/data_100x100.svm'
    else :
        file_path = 'datasets/mil_dataset/Musk/'+dataset_name +'norm.svm'

    #Load the data
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
        data_list.append([idi, idb, idc, feature_vector])#id of the instance, id of the bag, id of the class, feature vector

    data =[] #contains the list of bags
    num_bags = len(set([x[1] for x in data_list]))
    for i in range(num_bags):
        class_bag = [x[2] for x in data_list if x[1] == i][0]
        class_bag = 1 if class_bag == 1 else 0
        embedding = torch.tensor([x[3] for x in data_list if x[1] == i], dtype=torch.float32) #embedding of the bag size (nb_instances, nb_features)
        data.append((embedding,torch.tensor(class_bag)))

    return data

def get_c16_data(k=5):
    """
    Load data for Camelyon16 dataset and create the dataloaders for cross-validation
    Args:
        k (int): Number of folds for cross-validation
    Returns:
        list_train (list): List of tuples (path, label) for training
        list_test (list): List of tuples (path, label) for testing
        train_cross_val (list): List of training dataloaders for cross-validation
        val_cross_val (list): List of validation dataloaders for cross-validation
        y_test (list): List of labels for testing
    """
    normal_paths = pd.read_csv('datasets/Camelyon16/0-normal.csv')
    tumor_paths = pd.read_csv('datasets/Camelyon16/1-tumor.csv')

    list_test = [(path[0],path[1]) for path in normal_paths.values if 'test' in path[0]]+ [(path[0],path[1]) for path in tumor_paths.values if 'test' in path[0]]
    list_train = [(path[0],path[1]) for path in normal_paths.values if not 'test' in path[0]]+ [(path[0],path[1]) for path in tumor_paths.values if not 'test' in path[0]]        

    #Create the dataloaders for cross-validation
    train_cross_val = []
    val_cross_val = []
    y_tests =[]
    nb_train = len(list_train)
    size_fold = nb_train//k
    for i in range(k):

        train_subset = list_train[:i*size_fold] + list_train[(i+1)*size_fold:]
        val_subset = list_train[i*size_fold:(i+1)*size_fold]
        y_test = [item[1] for item in val_subset]
        #Create the dataloaders
        train_loader = torch.utils.data.DataLoader(
        dataset=train_subset,               
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=fn_collate
        )
        val_loader = torch.utils.data.DataLoader(
        dataset=val_subset,               
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=fn_collate
        )
        train_cross_val.append(train_loader)
        val_cross_val.append(val_loader)
        y_tests.append(y_test)
    return list_train,list_test,train_cross_val,val_cross_val,y_test

def fn_collate(batch):
    """
    Function to collate the data for the dataloaders
    """
    features =[torch.tensor(pd.read_csv(item[0]).values,dtype = torch.float32).to('cpu') for item in batch]
    labels = [torch.tensor(item[1]).to('cpu') for item in batch]
    return features[0], labels

# FUNCTION TO RUN #########################################

def run_TCGA(json):
    """
    Function to run the model on TCGA dataset
    Args:
        json (dict): Configuration file
    """
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
    
    #Change the embedding size of the model if needed
    json['model']['param']['embed_size'] = X_train[0].shape[1]
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
        if json['ROC'] == True:
            _, _, _, _,_, fpr, tpr = test(test_loader,y_test,model_class,print_results=False)
            #Save the results for the ROC curve in a csv file
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            name = model['name']
            df.to_csv(f'csv_files/TCGA/{name}.csv',index=False)
        else:
            test(test_loader,y_test,model_class,print_results=True)

        

def run_mil_dataset(json,dataset):
    """
    Function to run the model on MIL benchmark datasets
    Args:
        json (dict): Configuration file
        dataset (str): Name of the dataset
    """

    data = get_mil_data(dataset)
    X = [x[0] for x in data]
    y = [x[1] for x in data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset =[(X_train[i],y_train[i]) for i in range(len(X_train))]
    test_dataset =[(X_test[i],y_test[i]) for i in range(len(X_test))]

    #Change the embedding size of the model if needed
    json['model']['param']['embed_size'] = X_train[0].shape[1]

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
        if json['ROC'] == True:
            _, _, _, _,_, fpr, tpr = test(test_loader,y_test,model_class,print_results=False)
            #Save the results for the ROC curve in a csv file
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            name = model['name']
            df.to_csv(f'csv_files/TCGA/{name}.csv',index=False)
        else:
            test(test_loader,y_test,model_class,print_results=True)

def run_c16(json):
    """
    Function to run the model on Camelyon16 dataset
    Args:
        json (dict): Configuration file
    """
    
    list_train,list_test,train_cross_val,val_cross_val,y_test = get_c16_data(cross_param['k'])
    
    #Change the embedding size of the model if needed
    json['model']['param']['embed_size'] = 512
    if json['model']['name'] == 'dsmil':
        json['model']['i_classifier_param']['feature_size'] = 512
        json['model']['b_classifier_param']['input_size'] = 512

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
        cross_param = json['cross_val_param']
        k= cross_param['k']
        epochs = cross_param['epoch']
        k_fold_cross_validation_c16(train_cross_val,val_cross_val,y_test, model_class,  k, epochs)


    else : 
        train_param = json['training']
        epoch = train_param['epoch']
        lr = train_param['lr']
        weight_decay=train_param['weight_decay']
        batch_size = train_param['batch_size']

        print('----------Start Training----------')
        train_loader = torch.utils.data.DataLoader(dataset=list_train,batch_size=batch_size,shuffle=True,collate_fn=fn_collate)
        test_loader = torch.utils.data.DataLoader(dataset=list_test,batch_size=batch_size,shuffle=True,collate_fn=fn_collate)

        train(train_loader,epoch,model_class,lr, weight_decay,print_results=False)
        
        print('----------Start Testing----------')
        if json['ROC'] == True:
            _, _, _, _,_, fpr, tpr = test(test_loader,y_test,model_class,print_results=False)
            #Save the results for the ROC curve in a csv file
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            name = model['name']
            df.to_csv(f'csv_files/TCGA/{name}.csv',index=False)
        else:
            test(test_loader,y_test,model_class,print_results=True)




