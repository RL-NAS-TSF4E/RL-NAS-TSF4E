import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import gym

import utils
import models
from model_trainer import _process_one_batch, vali, train, _evaluate

import rl_nas

class RLNASEnv(gym.Env):
    def __init__(self, seq_len, label_len, pred_len, embed, freq, device, path, train_epochs, lradj, padding, inverse, features, train_loader, val_data, val_loader, default_val_mae):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.embed = embed
        self.freq = freq
        self.device = device
        self.path = path
        self.train_epochs = train_epochs
        self.lradj = lradj
        self.padding =  padding
        self.inverse = inverse 
        self.features = features
        self.train_loader = train_loader
        self.val_data = val_data
        self.val_loader = val_loader
        self.default_val_mae = default_val_mae

        self.observation_space = gym.spaces.MultiDiscrete([6, 6, 4, 6, 5, 3, 3])
        self.action_space = gym.spaces.MultiDiscrete([6, 6, 4, 6, 5, 3, 3])
        self.best_mae = default_val_mae
        self.best_trial = 0
        self.trial = 0
        self.initial_observation = np.array([1, 0, 1, 2, 4, 2, 0])
        self.actions_mae = []

    def reset(self):
        torch.cuda.empty_cache()
        return self.initial_observation

    def step(self, action):

        e_layers, d_layers, dropout_index, factor, n_heads, d_ff_index, d_model_index = action
        e_layers, d_layers, factor, n_heads = e_layers+1, d_layers+1, factor+3, n_heads+4
        dropout_values = [0.00, 0.05, 0.10, 0.15]
        dropout = dropout_values[dropout_index]
        d_ff_values = [512, 1024, 2048]
        d_ff = d_ff_values[d_ff_index]
        d_model_values = [512, 1024, 2048]
        d_model = d_model_values[d_model_index]

        #build modified model
        model = models.Informer(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=self.seq_len,
            label_len=self.label_len,
            out_len=self.pred_len,
            e_layers=e_layers,
            d_layers=d_layers,
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn='prob',
            embed=self.embed,
            freq=self.freq,
            activation='gelu',
            output_attention=False,
            distil=True,
            mix=True,
          ).float()

        model = model.to(self.device)


        #initiate parameters
        learning_rate = 0.0001
        patience=3
        train_steps = len(self.train_loader)
        early_stopping = utils.EarlyStopping(patience=patience, verbose=True)
        model_optim = optim.Adam(model.parameters(), lr=learning_rate)
        criterion =  nn.MSELoss()

        #train model
        train(model, self.train_loader, self.val_data, self.val_loader, criterion, model_optim, self.path, self.train_epochs, learning_rate, self.lradj, early_stopping, self.device, self.padding, self.pred_len, self.label_len , self.inverse, self.features)

        #evaluate on validation data
        preds, trues, mae, mse, rmse, mape, mspe = _evaluate(model, self.val_loader, self.val_data, _process_one_batch, self.device, self.padding, self.pred_len, self.label_len, self.inverse, self.features)

        torch.cuda.empty_cache()

        #aggregate actions & corresponding mae-values
        self.actions_mae.append([action, mae])

        #obs = self.observation_space.sample()
        obs = action
        reward = self.default_val_mae - mae
        done = True
        actions_mae = self.actions_mae
        info = {}

        #print result
        if mae > self.best_mae and self.best_mae == self.default_val_mae:
          print(f"[I {datetime.now()}] Trial {self.trial} finished with value: {mae} and parameters: 'e_layers': {e_layers} , 'd_layers': {d_layers}, 'dropout': {dropout}, 'factor': {factor}, 'n_heads': {n_heads}, ' d_ff: ' {d_ff}, 'd_model: '{d_model}. Best is trial default with value: {self.best_mae}.")
        if mae < self.best_mae:
          self.best_mae = mae
          self.best_trial = self.trial
          print(f"[I {datetime.now()}] Trial {self.trial} finished with value: {mae} and parameters: 'e_layers': {e_layers} , 'd_layers': {d_layers}, 'dropout': {dropout}, 'factor': {factor}, 'n_heads': {n_heads}, ' d_ff: ' {d_ff}, 'd_model: '{d_model}. Best is trial {self.best_trial} with value: {self.best_mae}.")
        if mae > self.best_mae and self.best_mae != self.default_val_mae:
          print(f"[I {datetime.now()}] Trial {self.trial} finished with value: {mae} and parameters: 'e_layers': {e_layers} , 'd_layers': {d_layers}, 'dropout': {dropout}, 'factor': {factor}, 'n_heads': {n_heads}, ' d_ff: ' {d_ff}, 'd_model: '{d_model}. Best is trial {self.best_trial} with value: {self.best_mae}.")
        self.trial = self.trial + 1

        return obs, reward, done, info, 
