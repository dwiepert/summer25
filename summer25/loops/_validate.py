"""
Validation loop

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import json
from pathlib import Path
from typing import Union
import time
##third party
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
##local
from summer25.models import HFModel

def validation(val_loader:DataLoader, model:Union[HFModel]):
    """
    """
    model.eval()

    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs= data['waveform'].to(model.device)

            outputs = model(inputs)

            loss = criterion(outputs)
            running_vloss += vloss.item()

    #TODO: Validation log

    return