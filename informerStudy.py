import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import optuna

import utils
import models
from model_trainer import _process_one_batch, vali, train, _evaluate

class InformerStudy:
    def __init__(self, seq_len, label_len, pred_len, embed, freq, device, path, train_epochs, lradj, padding, inverse, features, train_loader, val_data, val_loader, test_data, test_loader):
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
        self.test_data = test_data
        self.test_loader = test_loader
        self.actions_mae = []

    def objective(self, trial):
        e_layers = trial.suggest_int('e_layers', 1, 6)
        d_layers = trial.suggest_int('d_layers', 1, 6)
        dropout = trial.suggest_categorical('dropout', [0.0, 0.05, 0.10, 0.15])
        factor = trial.suggest_int('factor', 3, 8)
        n_heads = trial.suggest_int('n_heads', 4, 8)
        d_ff = trial.suggest_categorical('d_ff', [512, 1024, 2048])
        d_model = trial.suggest_categorical('d_model', [512, 1024, 2048])

        parameters = np.array([e_layers, d_layers, dropout, factor, n_heads, d_ff, d_model])

        #build model candidate
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
        preds, trues, val_mae, val_mse, val_rmse, val_mape, val_mspe = _evaluate(model, self.val_loader, self.val_data, _process_one_batch, self.device, self.padding, self.pred_len, self.label_len, self.inverse, self.features)

        torch.cuda.empty_cache()

        self.actions_mae.append([parameters, val_mae])

        return val_mae

    def run_study(self, n_trials=5):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        best_params = study.best_params

        e_layers = best_params['e_layers']
        d_layers = best_params['d_layers']
        dropout = best_params['dropout']
        factor = best_params['factor']
        n_heads = best_params['n_heads']
        d_ff = best_params['d_ff']
        d_model = best_params['d_model']

        #build best model
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

        #train best model
        train(model, self.train_loader, self.val_data, self.val_loader, criterion, model_optim, self.path, self.train_epochs, learning_rate, self.lradj, early_stopping, self.device, self.padding, self.pred_len, self.label_len , self.inverse, self.features)

        #evaluate on test data
        preds, trues, test_mae, test_mse, test_rmse, test_mape, test_mspe = _evaluate(model, self.test_loader, self.test_data, _process_one_batch, self.device, self.padding, self.pred_len, self.label_len, self.inverse, self.features)

        torch.save(model.state_dict(), self.path+'/'+'optuna_informer.pth')

        torch.cuda.empty_cache()

        return study, preds, trues, test_mae, test_mse, test_rmse, test_mape, test_mspe, self.actions_mae
