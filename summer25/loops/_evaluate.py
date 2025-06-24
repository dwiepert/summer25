"""
Evaluate a trained model

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import json
import math
from pathlib import Path
from typing import Union, Dict, List
##third party
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
##local
from summer25.models import HFModel
from emaae.utils import fro_norm3d, filter_encoding, filter_matrix, add_white_noise

def evaluate(test_loader:DataLoader, model:Union[HFModel]):
    """
    Evaluate model
    :param test_loader: test dataloader object
    :param model: trained model
    """

    out_dir = model.out_dir
    eval_dir = out_dir / 'eval'
    eval_dir.mkdir(exist_ok=True)

    model.eval()

    with torch.no_grad():
        inputs = data['waveform'].to(model.device)

        outputs = model(inputs)

        #TODO: EVAL METRICS

    #TODO: SAVE EVAL METRICS