import torch
import numpy as np
import utils

def _process_one_batch(model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, device, padding, pred_len, label_len, inverse, features):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # decoder input
    if padding == 0:
        dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    elif padding == 1:
        dec_inp = torch.ones([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    if inverse:
        outputs = dataset_object.inverse_transform(outputs)
    f_dim = -1 if features == 'MS' else 0
    batch_y = batch_y[:, -pred_len:, f_dim:].to(device)

    return outputs, batch_y

def vali(model, val_data, val_loader, criterion, device, padding, pred_len, label_len, inverse, features):
        model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(val_loader):
            pred, true = _process_one_batch(
                model, val_data, batch_x, batch_y, batch_x_mark, batch_y_mark, device, padding, pred_len, label_len, inverse, features)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        model.train()
        return total_loss

def train(model, train_loader, val_data, val_loader, criterion, model_optim, path, train_epochs, learning_rate, lradj, early_stopping, device, padding, pred_len, label_len , inverse, features):
    for epoch in range(train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1

            model_optim.zero_grad()
            pred, true = _process_one_batch(model, train_loader.dataset, batch_x, batch_y, batch_x_mark, batch_y_mark, device, padding, pred_len, label_len, inverse, features)
            loss = criterion(pred, true)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        vali_loss = vali(model, val_data, val_loader, criterion, device, padding, pred_len, label_len, inverse, features)
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        utils.adjust_learning_rate(model_optim, epoch + 1, learning_rate, lradj)

def _evaluate(model, data_loader, data, _process_one_batch, device, padding, pred_len, label_len, inverse, features):
    model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        pred, true = _process_one_batch(model, data, batch_x, batch_y, batch_x_mark, batch_y_mark, device, padding, pred_len, label_len, inverse, features)
        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse, rmse, mape, mspe = utils.metric(preds, trues)
    #print('MSE: {}, MAE: {}'.format(mse, mae))

    return preds, trues, mae, mse, rmse, mape, mspe
