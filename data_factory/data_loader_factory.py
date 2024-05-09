import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import data_factory

def get_data(flag, root_path='./ETDataset/ETT-small/', data_path='ETTh1.csv', seq_len=96, label_len=48, pred_len=24, features='M', target='OT', inverse=False, embed='timeF', batch_size=32, freq='h', num_workers=0, cols=None):
    timeenc = 0 if embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
    else:
        shuffle_flag = True
        drop_last = True

    data_set = data_factory.Dataset_ETT_hour(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        inverse=inverse,
        timeenc=timeenc,
        freq=freq,
        cols=cols
    )

    #print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
