
from torch.utils.data import Dataset
import numpy as np
from os.path import isfile, isdir
import json
import torch


class MyDataset(Dataset):
    def __init__(self, idx_to_file = None, data_folder = None):
        assert isfile(idx_to_file)
        assert isdir(data_folder)
        
        with open(idx_to_file, 'r') as f:
            self.maps = json.load(f)
        self.dir = data_folder

    def __len__(self):
        return len(self.maps.keys())

    def __getitem__(self, idx):
        return np.load('{}/{}'.format(self.dir, self.maps[str(idx)]))


if __name__ == '__main__':
    dataset = MyDataset(idx_to_file = 'idx_to_file.json', data_folder = 'dataset')
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])