import torch
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from src.MIL.ABMIL import *
from tqdm import tqdm

def train(train_loader,epoch,model,lr=0.001,weight_decay=0.0005,print_results=True):
    train_loss = 0.
    train_error = 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #betas ?

    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()
        
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.item()
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()
    
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    if print_results:
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))

def test(test_loader,y_test,model,print_results=True):
    model.eval()
    test_loss = 0.
    test_error = 0.
    y_pred =[]
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            if torch.cuda.is_available():
                data, bag_label = data.cuda(), bag_label.cuda()
            loss, _ = model.calculate_objective(data, bag_label)
            test_loss += loss.item()
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error
            y_pred.append(predicted_label.cpu().numpy().item())

            #print('Predicted label: {}, True label: {}'.format(predicted_label.item(), bag_label))
    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    if print_results:
        print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
        print('Accuracy :' , accuracy_score(y_test, y_pred))
        print('Precision :' , precision_score(y_test, y_pred))
        print('Recall :' , recall_score(y_test, y_pred))
        print('F1 Score :' , f1_score(y_test, y_pred))
    else:
        return test_error,f1_score(y_test, y_pred)


# Hyperparameter_tuning : 

#LIST OF HYPERPARAMETERS : 
# For ABMIL : hidden_size, dropout
# For VarMIL: hidden_size, dropout, gated, separate_attn, n_var_pools, act_func
# For CLAM : hidden_size, dropout, gated, instance_eval, subtyping, bag_weight
# On training : lr, weight_decay

hyp_ABMIL = {'hidden_size': [512, 1024], 'dropout': [0.1, 0.2, 0.3], 'lr': [0.001, 0.01], 'weight_decay': [0.0005, 0.001]}

def generate_hyperparameter_combinations(hyperparameters):

    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())
    
    # Générer les combinaisons
    combinations = product(*values)
    
    # Transformer les combinaisons en une liste de dictionnaires
    result = [dict(zip(keys, combination)) for combination in combinations]
    return result

def hyperparam_tuning(train_loader, test_loader, y_test,hyperparameters,model_class):
    best_error = 1
    best_f1 =0
    best_hyperparameters = {}
    combinations = generate_hyperparameter_combinations(hyperparameters)
    for hyperparameter in tqdm(combinations):
        model = model_class(**{key: value for key, value in hyperparameter.items() if key in model_class.__init__.__code__.co_varnames})
        for epoch in range(20):
            train(train_loader,epoch,model,lr=hyperparameter['lr'],weight_decay=hyperparameter['weight_decay'],print_results=False)
        test_error,f1 = test(test_loader,y_test,model,print_results=False)
        if f1 > best_f1:
            best_error = test_error
            best_f1 = f1
            best_hyperparameters = hyperparameter
    print('Best hyperparameters:', best_hyperparameters, 'Best error:', best_error, 'Best f1:', best_f1)


def neg_log_bernouilli(Y, pred_one): 
    return -1. * (Y * torch.log(pred_one) + (1. - Y) * torch.log(1-pred_one))