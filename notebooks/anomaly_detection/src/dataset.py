import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class NUSW_NB15_Dataset(Dataset):
    def __init__(self, path, mode='all'):
        ds_type = path.split('/')[-1].split('-')[0]
        df = pd.read_csv(path)
        
        if mode == 'normal':
            # get only normal data
            df = df[df['label'] == 0]
        elif mode == 'anomaly':
            # get only anormal data
            df = df[df['label'] == 1]
        
        x = df.drop(['id', 'attack_cat', 'label'], axis=1)
        y = df['label']

        self.x = torch.Tensor(x.to_numpy())
        self.y = torch.Tensor(y.to_numpy())

        self.dim = self.x.shape[1]

        print(
            f'Finished reading the {ds_type} set ({mode}) of Dataset '\
            f'({len(self.x)} samples found, each dim = {self.dim})'
        )

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    

def prep_dataloader(path, batch_size, shuffle, mode='all'):
    dataset = NUSW_NB15_Dataset(path, mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle
    )
    return dataloader