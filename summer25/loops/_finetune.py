"""
Finetuning loop

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
from torch.nn import BCELoss
from tqdm import tqdm
##local
from summer25.models import HFModel
from ._validate import validation

def finetune(train_loader:DataLoader, val_loader:DataLoader, model:Union[HFModel],
             optim_type:str="adamw", learning_rate:float=1e-3, loss_type:str="bce", scheduler_type:str=None,
             epochs:int=10, early_stop:bool=True,  patience:int=5, logging:bool=True, **kwargs):
    """
    Finetuning loop

    :param train_loader: DataLoader, train dataloader
    :param val_loader: DataLoader, validation dataloader
    :param model: initialized model
    :param optim_type: str, optimizer type to initialize (default=adamw)
    :param learning_rate: float, learning rate (default=1e-3)
    :param loss_type: str, loss to initialize (default=bce)
    :param scheduler_type: str, type of scheduler to initialize (default=None)
    :param epochs: int, number of epochs to train for (default=10)
    :param early_stop: boolean, indicate whether to use early stopping (default=True)
    :param patience: int, early stop patience (default = 5)
    :param logging: bool, True to save logs and checkpoints (default=True)
    :param kwargs: TODO
    :return model: finetuned
    """
    out_dir = model.out_dir
    log_path = out_dir / 'logs'
    logpath.mkdir(exist_ok=True)

    #optimizer
    if optim_type == 'adamw':
        optim = torch.optim.AdamW(params=model.parameters(),lr=learning_rate)
    else:
        return NotImplementedError(f'{optim_type} not implemented.')
    
    #loss
    if loss_type == 'BCE':
        criterion = BCELoss()
    else:
        return NotImplementedError(f'{loss_type} not implemented.')
    
    #scheduler
    if scheduler_type == 'exponential':
        assert 'end_lr' in kwargs, 'Must give end_lr if using exponential learning rate decay scheduler'
        end_lr = kwargs['end_lr']
        gamma = end_lr / (learning_rate**(1/epochs))
        print(f'LR scheduler gamma: {gamma}')
        scheduler = ExponentialLR(optim, gamma=gamma)
    
    #early stopping
    #TODO: print patience being used

    start_time = time.time()

    #FINETUNE LOOP
    for e in range(epochs):
        print('EPOCH {}:'.format(e + 1))
        est = time.time() #epoch start time

        model.train(True)
        running_loss = 0.
        for data in tqdm(train_loader):
            inputs = data['waveform'].to(model.device)
            optim.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs)
            loss.backward()
            running_loss += loss.item()

            optim.step()
        
        
        #TODO: Training log + DUMP log

        #TODO: validation loop, how frequently to run?
        validate(val_loader, model)

        #TODO: SAVING CHECKPOINTS w early stopping or just checkpointing
        

        if lr_scheduler is not None:
            lr_scheduler.step()

    #SAVE FINAL MODEL
    return model