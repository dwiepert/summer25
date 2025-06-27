"""
Custom model trainer

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import json
from pathlib import Path
from typing import Union, List
import time
##third party
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from tqdm import tqdm
##local
from summer25.models import HFModel
from ._early_stop import EarlyStopping
from ._ranked_clf_loss import RankedClassificationLoss

class Trainer():
    """
    Custom model training class

    :param model: initialized model
    :param target_features: list of target features
    :param optim_type: str, optimizer type (default = adamw)
    :param learning_rate: float, learning rate (default = 1e-3)
    :param loss_type: str, loss type (default = bce)
    :param scheduler_type: str, scheduler type (default = None)
    :param early_stop: bool, specify whether to use early stopping
    :param patience: int, patience for early stopping (default = 5)
    :param delta: float, minimum change for early stopping
    :param kwargs: additional values for rank classification loss or schedulers (e.g., rating_threshold/margin/bce_weight for rank loss and end_lr/epochs for Exponential scheduler)
    """
    def __init__(self, model:Union[HFModel], target_features:List[str], optim_type:str="adamw", 
                 learning_rate:float=1e-3, loss_type:str="bce", scheduler_type:str=None,
                 early_stop:bool=True,  patience:int=5, delta:float=0.0, **kwargs):
        self.model = model
        self.target_features = target_features
        self.learning_rate = learning_rate
        if optim_type == 'adamw':
            self.optim = torch.optim.AdamW(params=self.model.parameters(),lr=self.learning_rate)
        else:
            return NotImplementedError(f'{optim_type} not implemented.')
        
        #loss
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'rank':
            assert 'rating_threshold' in kwargs, 'Must give rating threshold for rank loss.'
            args = {'rating_threshold': kwargs.pop('rating_threshold')}
            if 'margin' in kwargs:
                args['margin'] = kwargs.pop('margin')
            if 'bce_weight' in kwargs:
                args['bce_weight'] = kwargs.pop('bce_weight')

            self.criterion = RankedClassificationLoss(**args)
        else:
            return NotImplementedError(f'{loss_type} not implemented.')
        
        #scheduler
        if scheduler_type == 'exponential':
            assert 'end_lr' in kwargs, 'Must give end_lr if using exponential learning rate decay scheduler'
            end_lr = kwargs['end_lr']
            epochs = kwargs['epochs']
            gamma = end_lr / (learning_rate**(1/epochs))
            print(f'LR scheduler gamma: {gamma}')
            self.scheduler = ExponentialLR(self.optim, gamma=gamma)
        else:
            self.scheduler = None
        
        #es 
        if early_stop:
            self.early_stop = EarlyStopping(patience=patience, delta=delta)
        else:
            self.early_stop = None

        self.log = {"train_loss":[], "val_loss":[]}


    def train_step(self, train_loader:DataLoader):
        """
        Training step
        :param train_loader: DataLoader with training data
        """
        self.model.train()
        running_loss = 0.
        for data in tqdm(train_loader):
            inputs, targets = data['waveform'].to(self.model.device), data['targets'].to(self.model.device)
            self.optim.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            running_loss += loss.item()

            self.optim.step()

        self.log['train_loss'].append(running_loss)
        

    def val_step(self, val_loader:DataLoader, e:int):
        """
        Validation step
        :param val_loader: Dataloader with validation data
        :param e: int, current epoch
        """
        self.model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader):
                inputs, targets= data['waveform'].to(self.model.device), data['targets'].to(self.model.device)
    
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                running_vloss += loss.item()

        self.log['val_loss'].append(running_vloss)

        if self.early_stop:
            self.early_stop(running_vloss, self.model, e)

        
    def fit(self, train_loader:DataLoader, val_loader:DataLoader=None, epochs:int=10):
        """
        Fit the model with train_loader and optional val_loader

        :param tain_loader: DataLoader with training data
        :param val_loader: DataLoader with validation data (default=None)
        :param epochs: int, number of epochs to train for
        """
        for e in range(epochs):
            self.train_step(train_loader)

            if val_loader:
                self.val_step(val_loader, e)

            #FLUSH LOG 
            with open(str(self.model.out_dir / 'train_log.json'), 'w') as f:
                json.dump(self.log, f)
            
            if self.early_stop:
                if self.early_stop.early_stop:
                    self.early_stop.best_model.save_model_components(f'best{self.early_stop.best_epoch}')
                    break
            
            #checkpointing
            if (e ==0 or e % 5 == 0) and e != epochs - 1:
                self.model.save_model_components(f'checkpoint{e}_')
            
            if self.scheduler:
                self.scheduler.step()
        
        if e == epochs - 1:
            self.model.save_model_components(f'final{e}_')

    def test(self, test_loader:DataLoader):
        """
        Evaluate model on test data

        :param testloader: DataLoader with test data
        """
        self.model.eval()
        running_loss = 0.0
        per_feature = {}
        for t in self.target_features:
            per_feature[t] = {'true':[], 'pred':[]}
        
        with torch.no_grad():
            running_loss = 0.0
            counter = 0
            for data in tqdm(test_loader):
                inputs, targets = data['waveform'].to(self.model.device), data['targets'].to(self.model.device)
                
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                for i in range(len(self.target_features)):
                    t = self.target_features[i]
                    temp = per_feature[t]
                    temp_true = temp['true']
                    temp_pred = temp['pred']

                    
                    outputs = outputs.cpu()
                    targets = targets.cpu()
                    temp_true.extend(targets[:,i].tolist())
                    temp_pred.extend([(o>0.5).float().item() for o in outputs[:,i]])
                    per_feature[t]= {'true':temp_true, 'pred':temp_pred}

        for t in self.target_features:
            temp = per_feature[t]
            true, pred = temp['true'],temp['pred']
            temp['bacc'] = balanced_accuracy_score(true, pred)
            temp['acc'] = accuracy_score(true, pred)
            temp['roc_auc'] = roc_auc_score(true, pred)
            per_feature[t] = temp

        metrics = {'loss':running_loss, 'feature_metrics': per_feature}
        with open(str(self.model.out_dir / 'evaluation.json'), 'w') as f:
            json.dump(metrics, f)
        

