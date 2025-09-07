# ML4Science - Benchmarking Multiple Instance Learning methods for Whole Slide Images

This repository hosts the implementation of a project part of the  **CS-433 Machine Learning** course at EPFL. This project was conducted in collaboration with the lab of Artificial Intelligence in Molecular Medecine. The objective of this project is to benchmark existings models for Multiple Instance Learning. 

## Dataset

This benchmark was conducted on three diferent datasets : a TCGA dataset , a dataset of classical MIL benchmarks and a the Clameyton 16 dataset. The classical MIL benchmark contains 5 datasets : Fox, Elephant, Tiger, Musk1 and Musk2. Musk 1 and Musk 2 are two datasets of molecules that are either musk or non-musk. The Fox, Elephant and Tiger datasets are datasets of parts of  images of animals. 
All the data corresponds to precomputed embeddings of the WSI images.
The TCGA dataset can be already found in the `datasets` folder, as it was provided by the AIMM lab. 
It contains the following files:
- `train_dict.pkl` : the training data
- `test_dict.pkl` : the testing data
Note that in our code we redo a train/test split on the complete dataset because the provided split was not balanced.

The other datasets can be downloaded using the following commands which will download the data and store it in the `datasets` folder.
```bash
python download.py --dataset mil
```
With the `--dataset` argument you can specify which dataset you want to download. The possible values are `mil` and `c16`.

Note that the Clameyton 16 dataset is heavy and needs some space to be downloaded.

## Repository Structure

```txt
.
├── csv_files                 # Contains the csv files (used to plot the ROC curves)
├── datasets                  # Stores the datasets used in the project.
├── json_files                # Contains the json files (used to store the best hyperparameters of the models)
├── requirements.txt          # List of dependencies for the project.
├── download.py               # Script to download the datasets.
├── main.py                   # Main script to run the project.
└── src
    ├── MIL                   # folder containing the implementation of the different MIL models
    └── utils.py              # Utilities for the project (test functions, hyperparameter tuning, etc.)
    └── run_fct.py            # Run functions for the different datasets

```

## Installation  

To use the code in this repository, follow these steps:

1. Clone the repository:

   ```bash  
   git clone https://github.com/CS-433/ml-project-2-judelru.git 
   ```

2. Create a virtual environment and install the dependencies (requirements.txt):

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset you want to use (see the Dataset section).

4. Run the main script:

   ```bash
   python main.py --dataset  --model
   ```

    The arguments are :
    - `--dataset` : the dataset you want to use. Possible values are `tcga`, `TCGA`, `Fox`, `Elephant`, `Tiger`, `musk1`, `musk2`, `C16`
    - `--model` : the name of the model you want to use. Use this argument if you don't want to change the hyperparameters. 
    - `--model_config` : the path to the json file containing the hyperparameters of the model (Look at the json_files folder for examples). Use this argument if you want to change the hyperparameters.
    - `--cross_val` : if you want to do cross validation. If you use this argument, the number of folds is specified in the json file. If it is set to false, the model will be trained on the training set and tested on the testing set.
    - `--ROC` : if you want to save the csv data for ROC curves.

## Contributors (Alphabetical Order)

- Julia Adonia Tillia Renard
- Duru Bektas
- Madeleine Hueber
