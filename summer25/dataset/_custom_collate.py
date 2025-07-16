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
    waveform = [item['waveform'] for item in batch]
    sample = {'uid':uid, 'targets':targets, 'sample_rate':sr, 'waveform':waveform}

    return sample
