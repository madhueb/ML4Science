import argparse
from pathlib import Path
from src.run_fct import *
import json

if __name__=="__main__":
    """
    File to run the MIL pipeline
    Args:
        model: str: Model to use, if None, the model configuration file must be specified
        model_config: Path: Path to the model configuration file
        dataset: str: Dataset to test the model on, default is TCGA (choices: TCGA, Fox, Elephant, Tiger, musk1, musk2, C16)
    """

    parser = argparse.ArgumentParser(description='MIL Pipeline')
    parser.add_argument('--model',type=str,choices=[None,'Emb_mean','Emb_max','Attention','GatedAttention','ACMIL','AttriMIL','CLAM','dsmil','TransMIL','VarMIL'],default=None,help='Model to use')
    parser.add_argument('--model_config', type=Path, default=None, help='Path to the model configuration file')
    parser.add_argument('--dataset',type = str,default = 'TCGA',help='Dataset to test the model on')
    parser.add_argument('--ROC',type = bool,default = None,help='Whether to save data for the ROC curve')
    
    args = parser.parse_args()

    if args.model is not None and args.model_config is None:
        path = 'json_files/'+args.model+'.json'
        with open(path, 'r') as f:
            model_config = json.load(f)
    elif args.model_config is not None:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        raise ValueError('Either a model or a model configuration file must be specified')

    if args.plot_ROC is not None:
        model_config['cross_val']= False
        model_config['ROC'] = args.plot_ROC

    if args.dataset == 'TCGA':
        data = run_TCGA(model_config)
    elif args.dataset in ['Fox','Elephant','Tiger','musk1','musk2']:
        data = run_mil_dataset(model_config,args.dataset)
    elif args.dataset == 'C16':
        data = run_c16(model_config)


