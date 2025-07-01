"""
Custom collate function

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
from typing import List
##third-party
import torch
##local
from summer25.transforms import Pad

def collate_features(batch: List[dict]) -> dict:
    """
    Collate function for use when model has a feature extractor that automatically extracts to same length features

    :param batch: input batch, list of samples (dictionaries)
    :return sample: output sample dict, samples collated appropriately
    """
    uid = [item['uid'] for item in batch]
    sr = [item['sample_rate'] for item in batch]
    targets = torch.stack([item['targets'] for item in batch])
    sample = {'uid':uid, 'targets':targets, 'sample_rate':sr, 'waveform':torch.concat([item['waveform'] for item in batch], dim=0), 'pad_tokens':[0 for item in batch]}

    return sample

def collate_waveform(batch:List[dict], pad:bool=False, pad_method:str='mean') -> dict:
    '''
    Collate function for waveforms - can specify padding

    :param batch: input batch, list of samples (dictionaries)
    :param pad: bool, indicate whether to pad waveforms within a batch
    :param pad_method: str, indicate whether to pad by 'mean' or 'zeros'
    :return sample: output sample dict, samples collated appropriately
    '''
    uid = [item['uid'] for item in batch]
    sr = [item['sample_rate'] for item in batch]
    targets = torch.stack([item['targets'] for item in batch])
    sample = {'uid':uid, 'targets':targets, 'sample_rate':sr, 'pad_tokens':[0 for item in batch]}

    if 'waveform' in batch[0]:
        #check 
        len = batch[0]['waveform'].size()
        all_same = [item['waveform'].size() == len for item in batch]
        if all(all_same):
            waveform = torch.stack([torch.squeeze(item['waveform']) for item in batch])
        elif pad:
            max_len = max([item['waveform'].shape[1] for item in batch])
            p = Pad(pad_method=pad_method)
            padded = [p(item, max_len) for item in batch]
            waveform = torch.stack([torch.squeeze(item['waveform']) for item in padded])
            pad_tokens = [item['pad_tokens'] for item in padded]
            sample['pad_tokens'] = pad_tokens
        else:
            waveform = [item['waveform'] for item in batch]
        sample['waveform'] = waveform
   
    return sample

def collate_wrapper(pad:bool, pad_method:str):
    """
    Wrapper for collate function
    :param pad: bool, indicate whether to pad waveforms within a batch
    :param pad_method: str, indicate whether to pad by 'mean' or 'zeros'
    :return: collate_waveform function with pad/pad_method given but batch not given 
    """
    return lambda b: collate_waveform(batch=b, pad=pad, pad_method=pad_method)