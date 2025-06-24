#IMPORTS
##built-in
import json
from pathlib import Path
from typing import Union
import time
##third party
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import BCELoss
from torchrl.record import CSVLogger
from tqdm import tqdm
##local
from summer25.models import HFModel
from ._validate import validation
from ._early_stop import EarlyStopping

class Trainer():
    """
    """
    def __init__(self, model,target_features, optim_type:str="adamw", learning_rate:float=1e-3, loss_type:str="bce", scheduler_type:str=None,
                 early_stop:bool=True,  patience:int=5, **kwargs):
        self.model = model
        self.target_features = target_features

        if optim_type == 'adamw':
            self.optim = torch.optim.AdamW(params=model.parameters(),lr=learning_rate)
        else:
            return NotImplementedError(f'{optim_type} not implemented.')
        
        #loss
        if loss_type == 'BCE':
            self.criterion = BCELoss()
        else:
            return NotImplementedError(f'{loss_type} not implemented.')
        
        #scheduler
        if scheduler_type == 'exponential':
            assert 'end_lr' in kwargs, 'Must give end_lr if using exponential learning rate decay scheduler'
            end_lr = kwargs['end_lr']
            gamma = end_lr / (learning_rate**(1/epochs))
            print(f'LR scheduler gamma: {gamma}')
            self.scheduler = ExponentialLR(optim, gamma=gamma)
        else:
            self.scheduler = None
        
        #es 
        if early_stop:
            self.early_stop = EarlyStopping(patience=patience)
        else:
            self.early_stop = None

        self.logger = CSVLogger(exp_name=self.model.model_name, log_dir=str(self.model.out_dir / 'logs'))

    def train_step(self, train_loader):
        """
        """
        self.model.train()
        running_loss = 0.
        for data in tqdm(train_loader):
            inputs, targets = data['waveform'].to(model.device), data['targets'].to(model.device)
            self.optim.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            running_loss += loss.item()

            self.optim.step()

        self.logger.log_scaler("train_loss", running_loss)

    def val_step(self, val_loader, e):
        """
        """
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                inputs, targets= data['waveform'].to(model.device), data['targets'].to(model.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                running_vloss += vloss.item()

        logger.log_scalar("val_loss", running_vloss)

        if self.early_stop:
            self.early_stop(running_vloss, self.model, e)

        
    def fit(self, train_loader, val_loader, epochs):
        """
        """
        start_time = time.time()
        for e in epochs:
            self.train_step(train_loader)

            if val_loader:
                self.val_step(val_loader, e)
            
            if self.early_stop:
                if self.early_stop.early_stop:
                    self.early_stop.best_model.save_model_components(f'best{self.early_stop.best_epoch}')
                    break
            
            #checkpointing
            if (e ==0 or e % 5 == 0) and e != epochs - 1:
                self.model.save_model_components(f'checkpoint{e}')
            
            if self.scheduler:
                self.scheduler.step()
        
        if e == epochs - 1:
            self.model.save_model_components(f'final{e}')

    def test(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        per_feature = {}
        for t in self.target_features:
            per_feature[t] = {'true':[], 'pred':[]}
        
        with torch.no_grad():
            running_loss = 0.0
            counter = 0
            for data in tqdm(test_loader):
                inputs, targets = data['waveform'].to(model.device), data['targets'].to(model.device)

                outputs = self.model(inputs)
                
                loss = self.criterion(inputs, targets)
                running_loss += loss.item()

                for i in range(len(self.target_features)):
                    t = self.target_features[i]
                    temp = per_feature[t]
                    temp_true = temp['true']
                    temp_pred = temp['pred']

                    outputs = outputs.cpu()
                    targets = targets.cpu()
                    temp_true.extend(targets[:,i].tolist())
                    temp_pred.extend(outputs[:,i].tolist())
                    per_feature[t]= {'true':temp_true, 'pred':temp_pred}

        per_feature_bacc = []
        per_feature_acc = []
        for t in self.target_features:
            temp = per_feature[t]
            true, pred = temp['true'],temp['pred']
            temp['bacc'] = balanced_accuracy_score(true, pred)
            temp['acc'] = accuracy_score(true, pred)
            temp['roc_auc'] = roc_auc_score(true, pred)
            per_feature[t] = temp

        
        #TODO: SAVE EVAL METRICS, see how the metrics are calculated and adapt accordingly

