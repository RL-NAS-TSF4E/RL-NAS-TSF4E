# Reinforcement Learning based Neural Architecture Search for Transformer guided Time Series Prediction Problems in the context of the Energy sector

**Master's Thesis**

**Author: Ali Jaabous** 

**1st Examiner: Prof. Dr. Stefan Lessmann** 

**2nd Examiner: Prof. Dr. Benjamin Fabian**

**Date: 21.05.2024**

![exp_approach](https://github.com/RL-NAS-TSF4E/RL-NAS-TSF4E/assets/168930273/1a890684-798f-4b33-946c-5e9ee2c2b70d)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

**Neural Architeture Search, Reinforcement Learning, Transfomer, Time Series Prediction**

As part of the thesis topic listed above, a comparison was made between a transformer model (Informer) optimized by Optuna and two instances of the model optimized with RL-NAS. This repository should make it possible to reproduce all results as well as offer the possibility of reproducibility.

## Working with the repo

### Dependencies

The project was realized with Visual Studio at the server grünau 9 of the computer science faculty of the Humboldt University. The Python version was 3.12.2

### Setup

1. Clone this repository and the rpository from https://github.com/zhouhaoyi/ETDataset/tree/main

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing results

After setup, the enclosed notebooks can be used to reproduce all experiments, results and visualizations.

All experiments and subdirectories can be realized using the notebook with the extension Exp. Specifically, the notebook covers the default case, the comparison experiment with Optuna and the two main experiments supported by RL-NAS. For this purpose, it is necessary to adapt the path specified in the notebook to your own infrastructure. Furthermore, the notebook generates the directories informer_checkpoints and agent_checkpoints within the code, in which the weights for the agents as well as the weights of the models optimized using the optimization approaches are stored. Due to their large size compremiered versions could only be made available. The pre-trained models can be loaded as required using model.load_state_dict(torch.load(path)). Of course, this requires initialization with the necessary parameters, which were specified for the corresponding approaches within the master's thesis.

All visualizations and metrics can be implemented using the notebook with the Viz extension. 

Both notebooks with executed cells will be added to this repo.

## Results

All results are stored in the results directory with the sub-directories for the respective approach.

## Project structure

The models and utils directories were taken from the original Informer repo and slightly adapted to the requirements of this project. 

The data_factory directory organizes data retrieval and data provision in the required format.

The class model_Trainer is responsible for the compliant training as well the evaluation and the testing

The directory rl_nas as well as the class informerStudy are organized as packages, which enable the use of the different optimization methods on the Informer architecture.

```bash
├── README.md
├── requirements.txt                                -- required libraries
├── RL_NAS_TSF4E_Exp.ipynb                          -- organizes the experiments and results as well the storage of the optimized models
├── RL_NAS_TSF4E_Viz.ipynb                          -- organizes all vizualisations
├── informerStudy.py                                -- package for application of optuna on the informer
├── model_trainer.py                                -- package which organizes the triaing as well the evaluation and testing
└── agent_checkpoints
    ├── a2c_model.pth                               -- stores weights of the a2c agent
    └── ppo_model.pth                               -- stores weights of the ppo agent
└── data_factory
    ├── data_loader.py                              -- preprocesses the dataset
    └── data_loader_factory.py                      -- builds dataset and data_loader
└── informer_checkpoints
    ├── a2c_informer.7z                              -- stores weights of the a2c optimized model
    ├── default_informer.7z                          -- stores weights of the default model
    ├── optuna_informer.7z                           -- stores weights of the optuna optimized model
    └── ppo_informer.7z                              -- stores weights of the ppo optimized model
└── models
    ├── __init__.py                                  -- wrapping classes into package
    ├── attn.py                                      -- implementation of the attention mechanism
    ├── decoder.py                                   -- decoder structure of the informer
    ├── embed.py                                     -- embedding function for the data processing
    ├── encoder.py                                   -- encoder structure of the informer
    └── model.py                                     -- calls the other classes and builds the informer
└── results
    ├── a2c
        ├── a2c_actions_mae.pkl                      -- architecture decisions & MAE during optimization
        ├── a2c_metrics.npy                          -- MAE, MSE, RMSE, MAPE, MSPE
        ├── a2c_pred.npy                             -- Predictions
        └── a2c_true.npy                             -- Groundtruth    
    ├── default
        ├── metrics.npy                              -- MAE, MSE, RMSE, MAPE, MSPE
        ├── pred.npy                                 -- Predictions
        └── true.npy                                 -- Groundtruth                              
    ├── optuna
        ├── optuna_actions_mae.pkl                   -- architecture decisions & MAE during optimization
        ├── optuna_metrics.npy                       -- MAE, MSE, RMSE, MAPE, MSPE
        ├── optuna_pred.npy                          -- Predictions
        └── optuna_true.npy                          -- Groundtruth                                      
    └── ppo
        ├── ppo_actions_mae.pkl                      -- architecture decisions & MAE during optimization
        ├── ppo_metrics.npy                          -- MAE, MSE, RMSE, MAPE, MSPE
        ├── ppo_pred.npy                             -- Predictions
        └── ppo_true.npy                             -- Groundtruth                                        
└── rl_nas
    ├── agents
        ├── __init__.py                              -- wrapping classes into package
        ├── a2c_agent.py                             -- implementation of the a2c agent
        └── ppo_agent.py                             -- implementation of the ppo agent    
    ├── environment
        ├── __init__.py                              -- wrapping classes into package 
        ├── rl_early_termination.py                  -- early termination function for the environment
        └── rl_nas_env.py                            -- implementation of an openAI based environment       
    └── __init__.py                                  -- wrapping classes into package                            
└── utils
    ├── __init__.py                                  -- wrapping classes into package
    ├── masking.py                                   -- helper function for the building of the informer 
    ├── metrics.py                                   -- helper function for the building of the informer 
    ├── timefeatures.py                              -- helper function for the building of the informer 
    └── tools.py                                     -- helper function for the building of the informer             
```
