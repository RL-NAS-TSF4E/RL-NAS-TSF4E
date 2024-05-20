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

[This is an example]

1. Clone this repository

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

Describe steps how to reproduce your results.

Here are some examples:
- [Paperswithcode](https://github.com/paperswithcode/releasing-research-code)
- [ML Reproducibility Checklist](https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/)
- [Simple & clear Example from Paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) (!)
- [Example TensorFlow](https://github.com/NVlabs/selfsupervised-denoising)

### Training code

Does a repository contain a way to train/fit the model(s) described in the paper?

### Evaluation code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

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
    ├── __init__.py                                  -- 
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
        ├── __init__.py                              -- 
        ├── a2c_agent.py                             -- 
        └── ppo_agent.py                             --     
    ├── environment
        ├── __init__.py                              -- 
        ├── rl_early_termination.py                  --
        └── rl_nas_env.py                            --         
    └── __init__.py                                  --                             
└── utils
    ├── __init__.py                                  -- 
    ├── index.txt                                    -- 
    ├── masking.py                                   -- 
    ├── metrics.py                                   -- 
    ├── timefeatures.py                              -- 
    └── tools.py                                     --              
```
