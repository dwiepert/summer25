"""
Custom model trainer

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
##built-in
import json
import gc
from pathlib import Path
import os
from typing import Union, List

##third party
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, SequentialLR, LambdaLR, CosineAnnealingLR
import torch.nn as nn
from tqdm import tqdm

##local
from summer25.models import HFModel
from summer25.io import upload_to_gcs
from ._early_stop import EarlyStopping
from ._ranked_clf_loss import RankedClassificationLoss

#HELPER FOR SCHEDULER
def warmup_lr_lambda(epoch:int, warmup_epochs:int = 5) -> float:
    """
    Function for warmup scheduler that defines when to stop warmup
    :param epoch: int, current epoch
    :param warmup_epochs: int, number of warmup epochs (default = 5)
    :return: fraction of warmup epochs completed
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

def warmup_wrapper(warmup_epochs:int = 5):
    """
    Wraper for warmup to flexibly set the number of warmup epochs
    
    :param warmup_epochs: int, number of warmup epochs (default = 5)
    :return: lambda function
    """
    return lambda w: warmup_lr_lambda(w, warmup_epochs=warmup_epochs)

#MAIN CLASS
class Trainer():
    """
    Custom model training class

    :param model: initialized model
    :param target_features: list of target features
    :param optim_type: str, optimizer type (default = adamw)
    :param learning_rate: float, learning rate (default = 1e-3)
    :param loss_type: str, loss type (default = bce)
    :param gradient_accumulation_steps: int, number of steps for gradient accumulation (default = 4)
    :param batch_size: int, batch size (default = 2)
    :param scheduler_type: str, scheduler type (default = None)
    :param early_stop: bool, specify whether to use early stopping (default = False)
    :param save_checkpoints: bool, specify whether to save checkpoints (default = True)
    :param patience: int, patience for early stopping (default = 5)
    :param delta: float, minimum change for early stopping (default = 0.0)
    :param kwargs: additional values for rank classification loss or schedulers (e.g., rating_threshold/margin/bce_weight for rank loss and end_lr/epochs for Exponential scheduler)
    """
    def __init__(self, model:Union[HFModel], target_features:List[str], optim_type:str="adamw", 
                 tf_learning_rate:float=None, learning_rate:float=1e-4, loss_type:str="bce", gradient_accumulation_steps:int=4, batch_size:int=2,
                 scheduler_type:str=None, early_stop:bool=False, save_checkpoints:bool=True, patience:int=5, delta:float=0.0, **kwargs):
        self.model = model
        self.name_prefix = f'{optim_type}_{loss_type}'
        self.target_features = target_features
        self.learning_rate= learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
    
        if tf_learning_rate:
            self.tf_learning_rate = tf_learning_rate
        else:
            self.tf_learning_rate = learning_rate
        self.name_prefix += f'_lr{self.learning_rate}_tflr{self.tf_learning_rate}'

        self.config = {'learning_rate': self.learning_rate, 'tf_learning_rate': self.tf_learning_rate, 'optim_type':optim_type, 'loss_type': loss_type, 'scheduler_type':scheduler_type,
                       'gradient_accumulation_steps':self.gradient_accumulation_steps, 'batch_size':self.batch_size}

        self.save_checkpoints = save_checkpoints
        if optim_type == 'adamw':
            self.tf_optim = torch.optim.AdamW(params=self.model.base_model.parameters(),lr=self.tf_learning_rate)
            self.clf_optim = torch.optim.AdamW(params=self.model.classifier_head.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f'{optim_type} not implemented.')
        
        #loss
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'rank':
            assert 'rating_threshold' in kwargs, 'Must give rating threshold for rank loss.'
            th = kwargs.pop('rating_threshold')
            args = {'rating_threshold': th}
            self.name_prefix += f'_th{th}'

            if 'margin' in kwargs:
                m = kwargs.pop('margin')
                args['margin'] = m
                self.name_prefix += f'_mar{m}'
            if 'bce_weight' in kwargs:
                b = kwargs.pop('bce_weight')
                args['bce_weight'] = b
                self.name_prefix += f'_weight{m}'

            self.config.update(args)
            self.criterion = RankedClassificationLoss(**args)
        else:
            raise NotImplementedError(f'{loss_type} not implemented.')
        
        #scheduler
        if scheduler_type == 'exponential':
            self.name_prefix += f'_{scheduler_type}'
            if 'gamma' in kwargs:
                clf_gamma = kwargs['gamma']
                if 'tf_gamma' in kwargs:
                    tf_gamma = kwargs['tf_gamma']
                else:
                    tf_gamma = clf_gamma
                self.name_prefix += f'_g{clf_gamma}_tg{tf_gamma}'
                self.config['gamma'] = clf_gamma
                self.config['tf_gamma'] = tf_gamma
            else:

                assert 'end_lr' in kwargs, 'Must give end_lr if using exponential learning rate decay scheduler.'
                assert 'epochs' in kwargs, 'Must give epochs if using exponential learning rate decay scheduler.'
                end_clf_lr = kwargs['end_lr']
                if 'end_tf_lr' in kwargs:
                    end_tf_lr = kwargs['end_tf_lr']
                else:
                    end_tf_lr = end_clf_lr
                epochs = kwargs['epochs']
                self.name_prefix += f'_endlr{end_clf_lr}_tfendlr{end_tf_lr}'
                tf_gamma = end_tf_lr / (self.tf_learning_rate**(1/epochs))
                clf_gamma = end_clf_lr / (self.learning_rate**(1/epochs))
                self.config['end_lr'] = end_clf_lr
                self.config['end_tf_lr'] = end_tf_lr
                self.config['epochs'] = epochs

            self.tf_scheduler = ExponentialLR(self.tf_optim, gamma=tf_gamma)
            self.clf_scheduler = ExponentialLR(self.clf_optim, gamma=clf_gamma)

        elif scheduler_type == 'warmup-cosine':
            assert 'warmup_epochs' in kwargs,'Must give warmup_epochs if using warmup.'
            assert 'train_len' in kwargs, 'Must give train_len if using cosine annealing lr.'
            clf_warmup_e = kwargs['warmup_epochs']
            if 'tf_warmup_epochs' in kwargs:
                tf_warmup_e = kwargs['tf_warmup_epochs']
            else:
                tf_warmup_e = clf_warmup_e
            self.config['warmup_epochs'] = clf_warmup_e
            self.config['tf_warmup_epochs'] = tf_warmup_e
            self.config['train_len'] = kwargs['train_len']

            self.name_prefix += f'_we{clf_warmup_e}_tfwe{tf_warmup_e}'
            tf_warmup = LambdaLR(self.tf_optim, lr_lambda=warmup_wrapper(tf_warmup_e))
            clf_warmup = LambdaLR(self.clf_optim, lr_lambda=warmup_wrapper(clf_warmup_e))
            tf_cosine = CosineAnnealingLR(self.tf_optim, T_max = kwargs['train_len'])
            clf_cosine = CosineAnnealingLR(self.clf_optim, T_max = kwargs['train_len'])
            self.tf_scheduler = SequentialLR(self.tf_optim, schedulers=[tf_warmup,tf_cosine], milestones=[tf_warmup_e-1]) #TODO: check what milestone should be based on warmup epoch?
            self.clf_scheduler = SequentialLR(self.clf_optim, schedulers=[clf_warmup,clf_cosine], milestones=[clf_warmup_e-1]) 
        
        elif scheduler_type == 'cosine':
            assert 'train_len' in kwargs, 'Must give train_len if using cosine annealing lr.'
            self.config['train_len'] = kwargs['train_len']
            self.tf_scheduler = CosineAnnealingLR(self.tf_optim, T_max = kwargs['train_len'])
            self.clf_scheduler = CosineAnnealingLR(self.clf_optim, T_max = kwargs['train_len'])
        
        elif scheduler_type is not None:
            raise NotImplementedError(f'{scheduler_type} not implemented.')
        else:
            self.tf_scheduler = None
            self.clf_scheduler = None
        
        #es 
        if early_stop:
            self.config['early_stop'] = True
            self.name_prefix += f'_es'
            es_params = {'patience':patience, 'delta':delta}
            self.config.update(es_params)
            if 'test' in kwargs:
                es_params['test'] = kwargs.pop('test')
            self.early_stop = EarlyStopping(**es_params)
        else:
            self.early_stop = None

        self.name_prefix += f'_bs{self.batch_size}_gas{self.gradient_accumulation_steps}'
        self.log = {"train_loss":[], "avg_train_loss":[], "val_loss":[], "avg_val_loss":[]}
        self.log.update(self.config)

    def train_step(self, train_loader:DataLoader):
        """
        Training step
        :param train_loader: DataLoader with training data
        """
        self.model.train()
        running_loss = 0.

        for index, data in enumerate(tqdm(train_loader)):
            self.model._check_memory(f'Batch {index} start.')

            inputs, attn_mask, targets = data['waveform'].to(self.model.device), data['attn_mask'].to(self.model.device), data['targets'].to(self.model.device)
            
            #self.tf_optim.zero_grad()
            #self.clf_optim.zero_grad()

            outputs = self.model(inputs, attn_mask)

            loss = self.criterion(outputs, targets)
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()
            self.model._check_memory('Backprop finished.')

            if (index + 1) % self.gradient_accumulation_steps == 0:
                self.tf_optim.step()
                self.clf_optim.step()
                self.tf_optim.zero_grad()
                self.clf_optim.zero_grad()
            
            del targets
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

            self.model._check_memory('Cache emptied.')


        self.log['train_loss'].append(running_loss)
        self.log['avg_train_loss'].append((running_loss / len(train_loader)))
        
        self.fit = False

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
                self.model._check_memory('Validation start.')
                
                inputs, attn_mask, targets = data['waveform'].to(self.model.device), data['attn_mask'].to(self.model.device), data['targets'].to(self.model.device)
    
                outputs = self.model(inputs, attn_mask)

                loss = self.criterion(outputs, targets)
                running_vloss += loss.item()

                del targets
                del outputs
                gc.collect()
                torch.cuda.empty_cache()


        self.log['val_loss'].append(running_vloss)
        self.log['avg_val_loss'].append((running_vloss / len(val_loader)))

        if self.early_stop:
            self.early_stop(running_vloss, self.model, e)

        
    def fit(self, train_loader:DataLoader, val_loader:DataLoader=None, epochs:int=10):
        """
        Fit the model with train_loader and optional val_loader

        :param tain_loader: DataLoader with training data
        :param val_loader: DataLoader with validation data (default=None)
        :param epochs: int, number of epochs to train for
        """
        self.epochs = epochs
        name_prefix =  f'{self.name_prefix}_e{epochs}'
        #FLUSH CONFIG
        if self.model.bucket:
            self.path = Path('.')
            self.upload_path = self.model.out_dir / name_prefix
        else:
            self.path = self.model.out_dir / name_prefix
            self.path.mkdir(parents = True, exist_ok = True)

        self.model.save_config(sub_dir=name_prefix)

        config_path = self.path / 'train_config.json'
        with open(str(config_path), 'w') as f:
            json.dump(self.config, f)

        if self.model.bucket:
            upload_to_gcs(self.upload_path, config_path, self.model.bucket, overwrite=True)
            os.remove(config_path)

        for e in range(epochs):
            self.tf_optim.zero_grad()
            self.clf_optim.zero_grad()

            self.train_step(train_loader)

            if val_loader:
                self.val_step(val_loader, e)

            log_path = self.path / 'train_log.json'
            with open(str(log_path), 'w') as f:
                json.dump(self.log, f)

            if self.model.bucket:
                upload_to_gcs(self.upload_path, log_path, self.model.bucket, overwrite=True)
                os.remove(log_path)
            
            if self.early_stop:
                if self.early_stop.early_stop:
                    best_model, best_epoch, _ = self.early_stop.get_best_model()
                    out_name = Path(name_prefix) /f'best{best_epoch}'
                    best_model.save_model_components(sub_dir=out_name)
                    break
            
            #checkpointing
            if (e ==0 or e % 5 == 0) and (e != epochs - 1) and self.save_checkpoints:
                out_name = Path(name_prefix) /f'checkpoint{e}'
                self.model.save_model_components(sub_dir=out_name)
            
            if self.tf_scheduler:
                self.tf_scheduler.step()
                self.clf_scheduler.step()
        
            if e == epochs - 1:
                out_name = Path(name_prefix) /f'final{e}'
                self.model.save_model_components(sub_dir = out_name)

            self.fit = True 

    def test(self, test_loader:DataLoader):
        """
        Evaluate model on test data

        :param testloader: DataLoader with test data
        """
        if not self.fit:
            name_prefix = self.name_prefix
            if self.model.bucket:
                self.path = Path('.')
                self.upload_path = self.model.out_dir / name_prefix
            else:
                self.path = self.model.out_dir / name_prefix
                self.path.mkdir(parents = True, exist_ok = True)
                self.model.save_config(sub_dir=name_prefix)
            
        self.model.eval()
        running_loss = 0.0
        per_feature = {}
        for t in self.target_features:
            per_feature[t] = {'true':[], 'pred':[]}
        
        with torch.no_grad():
            running_loss = 0.0
            for data in tqdm(test_loader):
                inputs, attn_mask, targets = data['waveform'].to(self.model.device), data['attn_mask'].to(self.model.device), data['targets'].to(self.model.device)
                
                outputs = self.model(inputs, attn_mask)
                
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                outputs = outputs.cpu()
                targets = targets.cpu()
                for i in range(len(self.target_features)):
                    t = self.target_features[i]
                    temp = per_feature[t]
                    temp_true = temp['true']
                    temp_pred = temp['pred']
                    temp_true.extend(targets[:,i].tolist())
                    temp_pred.extend([(o>0.5).float().item() for o in outputs[:,i]])
                    per_feature[t]= {'true':temp_true, 'pred':temp_pred}
                
                try:
                    del targets 
                    del outputs
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass


        for t in self.target_features:
            temp = per_feature[t]
            true, pred = temp['true'],temp['pred']
            temp['bacc'] = balanced_accuracy_score(true, pred)
            temp['acc'] = accuracy_score(true, pred)
            #temp['roc_auc'] = roc_auc_score(true, pred)
            per_feature[t] = temp

        metrics = {'loss':running_loss, 'avg_loss': (running_loss / len(test_loader)), 'feature_metrics': per_feature}
        
        log_path = self.path / 'evaluation.json'
        
        with open(str(self.path / 'evaluation.json'), 'w') as f:
            json.dump(metrics, f)
        
        if self.model.bucket:
            upload_to_gcs(self.upload_path, log_path, self.model.bucket, overwrite=True)
            os.remove(log_path)