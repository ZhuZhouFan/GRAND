import os
import torch
from torch.utils.data import Dataset
import numpy as np

class SGA_Dataset(Dataset):
    def __init__(self, data_dir, start_time, end_time):
        self.data_info = self.get_data_info(data_dir, start_time, end_time)

    def __getitem__(self, index):
        data_path = self.data_info[index]
        feature = np.load(f'{data_path}/feature.npy')
        label = np.load(f'{data_path}/label.npy')
        feature = torch.Tensor(feature)
        label = torch.Tensor(label)
        return feature, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_data_info(data_dir, start_time, end_time):
        data_info = list()
        date_list = os.listdir(data_dir)
        date_list.sort()
        valid_date_list = list(filter(lambda x: (x >= start_time) & (x <= end_time), date_list))
        for date in valid_date_list:
            data_info.append(f'{data_dir}/{date}')
        return data_info