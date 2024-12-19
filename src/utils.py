import torch
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from scipy import stats
import numpy as np


######################################## FUNCTIONS TO TRAIN AND TEST MODELS ########################################

def train(train_loader,epoch,model,lr=0.001,weight_decay=0.0005,print_results=True):
    """
    Train the model for one epoch
    Args:
        train_loader (DataLoader): DataLoader for the training set
        epoch (int): Current epoch number
        model (nn.Module): Model to train
        lr (float): Learning rate, default is 0.001
        weight_decay (float): Weight decay, default is 0.0005
        print_results (bool): Whether to print the results, default is True
    """
    model.train()
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
        error, _, _ = model.calculate_classification_error(data, bag_label)
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

def test(test_loader,y_test,model,print_results=False):
    """
    Test the model
    Args:
        test_loader (DataLoader): DataLoader for the test set
        model (nn.Module): Model to test
        print_results (bool): Whether to print the results, default is False
    """
    model.eval()
    test_loss = 0.
    test_error = 0.
    y_pred =[]
    y_probs=[]
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            if torch.cuda.is_available():
                data, bag_label = data.cuda(), bag_label.cuda()
            loss, _ = model.calculate_objective(data, bag_label)
            test_loss += loss.item()
            error, predicted_label, y_prob = model.calculate_classification_error(data, bag_label)
            test_error += error
            y_pred.append(predicted_label.cpu().numpy().item())
            y_probs.append(y_prob[0])

            #print('Predicted label: {}, True label: {}'.format(predicted_label.item(), bag_label))
    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    
    
    if print_results:
        print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
        print('F1 Score :', f1)
        print('Accuracy :', accuracy)
        print('Precision :', precision)
        print('Recall :', recall)      
    else:
        return test_error, f1, accuracy, precision, recall, fpr, tpr



def k_fold_cross_validation(train_dataset, model_class, k=5, epochs=20, lr=0.001, weight_decay=0.0005, batch_size=1):
    """
    Perform k-fold cross-validation on the train_dataset
    Args:
        train_dataset (Dataset) : Dataset to perform cross-validation on
        model_class (nn.Module): Model to train
        k (int): Number of folds, default is 5
        epochs(int): Number of epochs, default is 20
        lr (float): Learning rate, default is 0.001
        weight_decay (float): Weight decay, default is 0.0005
        batch_size (int): Batch size, default is 1

    Prints the average test error, f1 score, accuracy, precision and recall over the k folds

    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
 

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f'Fold {fold + 1}/{k}')
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # extract labels for validation set
        y_val = [train_dataset[idx][1].item() for idx in val_idx]

        model = model_class
        if torch.cuda.is_available():
            model.cuda()

        for epoch in range(1, epochs + 1):
            train(train_loader, epoch, model, lr, weight_decay)

        print(f'Evaluating Fold {fold + 1}')
        test_error, f1, accuracy, precision, recall, _, _ = test(val_loader, y_val, model, print_results=False)
        fold_results.append((test_error, f1, accuracy, precision, recall))
     
    errors = [ r[0] for r in fold_results]
    f1_scores = [ r[1] for r in fold_results ]
    accuracies = [ r[2] for r in fold_results]
    precisions = [ r[3] for r in fold_results]
    recalls = [ r[4] for r in fold_results]

    print(f'\nK-Fold Cross-Validation Results:')
    print(f'Average Test Error: {np.mean(errors):.4f} +/- {stats.sem(errors):.4f}')
    print(f'Average F1 Score: {np.mean(f1_scores):.4f} +/- {stats.sem(f1_scores):.4f}')
    print(f'Average Accuracy: {np.mean(accuracies):.4f} +/- {stats.sem(accuracies):.4f}')
    print(f'Average Precision: {np.mean(precisions):.4f} +/- {stats.sem(precisions):.4f}')
    print(f'Average Recall: {np.mean(recalls):.4f} +/- {stats.sem(recalls):.4f}')



    return fold_results

def k_fold_cross_validation_c16(train_cross_val,val_cross_val,y_tests, model_class, k=5, epochs=20, lr=0.001, weight_decay=0.0005, batch_size=1):
    """
    Perform k-fold cross-validation for the Camelyon16 dataset, where the train and validation sets are already split
    Args:
        train_cross_val (list): List of DataLoaders for the training set for each fold
        val_cross_val (list): List of DataLoaders for the validation set for each fold
        y_tests (list): List of labels for the validation set
        model_class (nn.Module): Model to train
        k (int): Number of folds, default is 5
        epochs(int): Number of epochs, default is 20
        lr (float): Learning rate, default is 0.001
        weight_decay (float): Weight decay, default is 0.0005
        batch_size (int): Batch size, default is 1

    Prints the average test error, f1 score, accuracy, precision and recall over the k folds
    """
    
    fold_results = []
    for fold in range(k):
        print(f'Fold {fold + 1}/{k}')
        train_loader = train_cross_val[fold]
        val_loader = val_cross_val[fold]

        # extract labels for validation set
        y_val = y_tests[fold]

        model = model_class
        for epoch in range(1, epochs + 1):
            train(train_loader, epoch, model, lr, weight_decay)

        print(f'Evaluating Fold {fold + 1}')
        test_error, f1, accuracy, precision, recall = test(val_loader, y_val, model, print_results=False)
        fold_results.append((test_error, f1, accuracy, precision, recall))
     
    errors = [r[0].cpu().numpy() if isinstance(r[0], torch.Tensor) else r[0] for r in fold_results]
    f1_scores = [r[1].cpu().numpy() if isinstance(r[1], torch.Tensor) else r[1] for r in fold_results]
    accuracies = [r[2].cpu().numpy() if isinstance(r[1], torch.Tensor) else r[2] for r in fold_results]
    precisions = [r[3].cpu().numpy() if isinstance(r[1], torch.Tensor) else r[3] for r in fold_results]
    recalls = [r[4].cpu().numpy() if isinstance(r[1], torch.Tensor) else r[4] for r in fold_results]

    print(f'\nK-Fold Cross-Validation Results:')
    print(f'Average Test Error: {np.mean(errors):.4f} +/- {stats.sem(errors):.4f}')
    print(f'Average F1 Score: {np.mean(f1_scores):.4f} +/- {stats.sem(f1_scores):.4f}')
    print(f'Average Accuracy: {np.mean(accuracies):.4f} +/- {stats.sem(accuracies):.4f}')
    print(f'Average Precision: {np.mean(precisions):.4f} +/- {stats.sem(precisions):.4f}')
    print(f'Average Recall: {np.mean(recalls):.4f} +/- {stats.sem(recalls):.4f}')



    return fold_results


######################################## FUNCTIONS FOR HYPERPARAMETER TUNING ######################################## 

def generate_hyperparameter_combinations(hyperparameters):
    """
    Generate all possible combinations of hyperparameters
    Args:
        hyperparameters (dict): Dictionary containing hyperparameters and their possible values
    Returns:
        List of dictionaries, each containing a combination of hyperparameters
    """

    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())
    
    # Create all possible combinations of hyperparameters
    combinations = product(*values)
    
    # Combine all combinations into a list of dictionaries
    result = [dict(zip(keys, combination)) for combination in combinations]
    return result

def hyperparam_tuning(train_loader, test_loader, y_test,hyperparameters,model_class):
    """
    Perform hyperparameter tuning using grid search for f1 score
    Args:
        train_loader (DataLoader): DataLoader for the training set
        test_loader (DataLoader): DataLoader for the test set
        y_test (list): List of labels for the test set
        hyperparameters (dict): Dictionary containing hyperparameters and their possible values
        model_class (nn.Module): Model to train

    Prints the best hyperparameters and the corresponding test error and f1 score
    """
    best_error = 1
    best_f1 =0
    best_hyperparameters = {}
    combinations = generate_hyperparameter_combinations(hyperparameters)
    for hyperparameter in tqdm(combinations):
        if isinstance(model_class, dict) : 
            i_classifier = model_class['i_classifier']
            b_classifier = model_class['b_classifier'](
            **{key: value for key, value in hyperparameter.items() 
               if key in model_class['b_classifier'].__init__.__code__.co_varnames})
            model = model_class['dsmil'](
            i_classifier=i_classifier,
            b_classifier=b_classifier,
            **{key: value for key, value in hyperparameter.items() 
               if key in model_class['dsmil'].__init__.__code__.co_varnames}
        )
        else : 
            model = model_class(**{key: value for key, value in hyperparameter.items() if key in model_class.__init__.__code__.co_varnames})
        for epoch in range(20):
            train(train_loader,epoch,model,lr=hyperparameter['lr'],weight_decay=hyperparameter['weight_decay'],print_results=False)
        test_error,f1 = test(test_loader,y_test,model,print_results=False)
        if f1 > best_f1:
            best_error = test_error
            best_f1 = f1
            best_hyperparameters = hyperparameter
    print('Best hyperparameters:', best_hyperparameters, 'Best error:', best_error, 'Best f1:', best_f1)

######################################## FUNCTIONS FOR LOSS FUNCTIONS ########################################
def neg_log_bernouilli(Y, pred_one): 
    return -1. * (Y * torch.log(pred_one) + (1. - Y) * torch.log(1-pred_one))



