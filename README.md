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
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

(Short summary of motivation, contributions and results)

**Keywords**: xxx (give at least 5 keywords / phrases).

**Full text**: [include a link that points to the full text of your thesis]
*Remark*: a thesis is about research. We believe in the [open science](https://en.wikipedia.org/wiki/Open_science) paradigm. Research results should be available to the public. Therefore, we expect dissertations to be shared publicly. Preferably, you publish your thesis via the [edoc-server of the Humboldt-Universität zu Berlin](https://edoc-info.hu-berlin.de/de/publizieren/andere). However, other sharing options, which ensure permanent availability, are also possible. <br> Exceptions from the default to share the full text of a thesis require the approval of the thesis supervisor.  

## Working with the repo

### Dependencies

Which Python version is required? 

Does a repository have information on dependencies or instructions on how to set up the environment?

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

All experiments and subdirectories can be realized using the notebook with the extension Exp. Specifically, the notebook covers the default case, the comparison experiment with Optuna and the two main experiments supported by RL-NAS. Furthermore, the notebook generates the directories informer_checkpoints and agent_checkpoints within the code, in which the weights for the agents as well as the weights of the models optimized using the optimization approaches are stored. All results are stored in the results directory with the results for the respective approach. 

### Training, Evaluation & Testing



### Pretrained models

Does a repository provide free access to pretrained model weights?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Project structure

```bash
├── README.md
├── requirements.txt                                -- required libraries
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
