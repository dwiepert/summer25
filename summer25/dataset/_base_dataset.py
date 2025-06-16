"""
Base dataset class

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
#built-in
from typing import List, Union 

#third-party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    '''
    Simple audio dataset
    '''
    
    def __init__(self, annotations_df:Union[pd.DataFrame, np.ndarray], target_labels:List[str], transforms=None):
        '''
        Initialize dataset with dataframe, target labels, and list of transforms
        :param annotations_df: pd.DataFrame or numpy array, table with all annotation values
        :param target_labels: List[str], list of target columns in annotations_df
        :param transforms: torchvision transforms function to run on data (default=False)
        '''
        super(BaseDataset, self).__init__()

        self.annotations_df = annotations_df
        self.transforms= transforms
        self.target_labels = target_labels
        
    def __len__(self) -> int:
        '''
        Get dataset size
        :return: int, len of dataset
        '''
        return len(self.annotations_df)
    
    def __getitem__(self, idx:Union[torch.Tensor, int, str, List[str], List[int]]) -> dict:
        '''
        Run transformation
        :return: dict, transformed data sample
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        uid = self.annotations_df.index[idx]
        targets = self.annotations_df[self.target_labels].iloc[idx].values
        
        sample = {
            'uid' : uid,
            'targets' : targets
        }

        if self.transforms is not None:
            return self.transforms(sample)
        else:
            return sample