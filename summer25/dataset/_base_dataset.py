"""
Base dataset class

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
#built-in
from typing import List, Union 

#third-party
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    '''
    Simple audio dataset
    '''
    
    def __init__(self, data:Union[pd.DataFrame], target_labels:Union[List[str]], transforms=None):
        '''
        Initialize dataset with dataframe, target labels, and list of transforms
        :param data: pd.DataFrame, table with all annotation values
        :param target_labels: List[str], list of target columns in data
        :param transforms: torchvision transforms function to run on data (default=None)
        '''
        super(BaseDataset, self).__init__()

        self.data = data.copy()
        self.transforms= transforms
        self.target_labels = target_labels
        
        assert isinstance(self.data, pd.DataFrame), 'Must give dataframe'
        assert self.target_labels is not None, 'Must give target labels.'
        if isinstance(self.data, pd.DataFrame):
            assert all([isinstance(v,str) for v in self.target_labels]), 'Must give string column names if using a data frame.'
            assert set(self.target_labels).issubset(set(self.data.columns.to_list())), 'Given target labels not in data'
        else:
            raise TypeError('Must give annotations as either pandas dataframe or numpy array')

    def get_data(self, only_targets:bool=False) -> pd.DataFrame:
        '''
        Return dataframe
        :param only_targets:bool, specify whether to only get target label columns from data
        :return self.data: pd.DataFrame
        '''
        if only_targets:
            return self.data[self.target_labels]
        
        return self.data
    
    def __len__(self) -> int:
        '''
        Get dataset size
        :return: int, len of dataset 
        '''
        return len(self.data)
    
    def __getitem__(self, idx:Union[int, torch.Tensor, List[int]]) -> dict:
        '''
        Run transformation
        :param idx: index as int
        :return: dict, transformed data sample, Expects that the uid column has been set to index
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list): assert len(idx) == 1, 'Should only have one idx even if given as a tensor.'
        
        uid = self.data.index[idx]

        targets = self.data[self.target_labels].iloc[idx].values.tolist()
        
        sample = {
            'uid' : uid,
            'targets' : targets
        }

        if self.transforms is not None:
            return self.transforms(sample)
        else:
            return sample