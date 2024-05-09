import torch
import torch.nn as nn
import torch.optim as optim

import utils
import models
from model_trainer import _process_one_batch, vali, train, _evaluate

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

import rl_nas

class A2CAgent:
    def __init__(self, seq_len, label_len, pred_len, embed, freq, device, path, train_epochs, lradj, padding, inverse, features, train_loader, val_data, val_loader, env):
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
        self.env = env

    def get_a2c_results(self, steps):
        # Wrap environment in DummyVecEnv
        env = DummyVecEnv([lambda: self.env])

        # Define A2C model
        a2c_model = A2C("MlpPolicy", env, verbose=1)

        # Train the model
        a2c_model.learn(total_timesteps=steps, callback=rl_nas.environment.StopTrainingAfterNStepsCallback(n_steps=steps))

        model_path = ( "./agent_checkpoints/a2c_model.pth")
        a2c_model.save(model_path)

        torch.cuda.empty_cache()

        # Retrieve actions_mae from the environment
        a2c_actions_mae = env.envs[0].actions_mae

        # Find the best parameters
        min_mae = min(item[1] for item in a2c_actions_mae)
        best_parameters = next(item for item in a2c_actions_mae if item[1] == min_mae)
        best_parameters = best_parameters[0]

        return a2c_actions_mae, best_parameters

    def get_best_model(self, best_parameters):
        e_layers, d_layers, dropout_index, factor, n_heads, d_ff_index, d_model_index = best_parameters
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

        torch.save(model.state_dict(), self.path+'/'+'a2c_informer.pth')

        return preds, trues, mae, mse, rmse, mape, mspe
