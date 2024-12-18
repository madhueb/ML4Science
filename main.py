import argparse
from pathlib import Path
from src.run_fct import *
import json

if __name__=="__main__":

    
    parser = argparse.ArgumentParser(description='MIL Pipeline')
    parser.add_argument('--model',type=str,choices=[None,'Emb_mean','Emb_max','Attention','GatedAttention','ACMIL','AttriMIL','CLAM','dsmil','TransMIL','VarMIL'],default=None,help='Model to use')
    parser.add_argument('--model_config', type=Path, required=True, help='Path to the model configuration file')
    parser.add_argument('--dataset',type = str,default = 'TCGA',help='Dataset to test the model on')
    
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


    if args.dataset == 'TCGA':
        data = run_TCGA(model_config)
    elif args.dataset in ['Fox','Elephant','Tiger','musk1','musk2']:
        data = run_mil_dataset(model_config,args.dataset)
    elif args.dataset == 'C16':
        data = run_c16(model_config)


