"""
Base model class for setting up audio models from various sources

Author(s): Daniela Wiepert
Last modified: 06/2025
"""

#IMPORT
##built-in
from abc import abstractmethod
from collections import OrderedDict
import json
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

##third-party
import numpy as np
import torch
import torch.nn as nn

##local
from ._classifier import Classifier

class BaseModel(nn.Module):
    """
    Base model class
    Includes all shared parameters

    :param out_dir: pathlike, path to directory for saving all model information
    :param clf_args: dict of Classifier args (see README for examples, should include clf ckpt if that is trained)
    :param freeze_method: str, freeze method for base pretrained model (default=all)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param pt_ckpt: pathlike, path to base pretrained model checkpoint (default=None)
    :param ft_ckpt: pathlike, path to finetuned base model checkpoint (default=None)
    :param kwargs: additional arguments for optional parameters (e.g., pool_dim for mean/max pooling and unfreeze_layers if freeze_method is layer)
    """
    def __init__(self, out_dir:Union[Path, str], clf_args:Dict, freeze_method:str = 'all', pool_method:str = 'mean',
                 pt_ckpt:Optional[Union[Path, str]]=None, ft_ckpt:Optional[Union[Path,str]]=None, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs):
        
        # INITIALIZE VARIABLES
        self.out_dir = out_dir
        if not isinstance(self.out_dir, Path): self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.clf_args = clf_args
        self.pt_ckpt=pt_ckpt
        self.ft_ckpt=ft_ckpt
        self.freeze_method = freeze_method
        self.pool_method = pool_method

        # ASSERTIONS
        if self.pt_ckpt is not None:
            if not isinstance(self.pt_ckpt, Path): self.pt_ckpt = Path(self.pt_ckpt)
            assert self.pt_ckpt.exists(), f'Pretrained model path {self.pt_ckpt} does not exist.'
        if self.ft_ckpt is not None:
            if not isinstance(self.ft_ckpt, Path): self.ft_ckpt = Path(self.ft_ckpt)
            assert self.ft_ckpt.exists(), f'Finetuned model path {self.ft_ckpt} does not exist.'
        assert self.freeze_method in ['all', 'layer', 'none'], f'{self.freeze_method} is not a valid freeze method. Use one of [`all`, `layer`, `none`].'
        if self.freeze_method == 'layer':
            assert 'unfreeze_layers' in kwargs, 'Layers to unfreeze not given as input'
            self.unfreeze_layers = kwargs['freeze_layers']
        assert self.pool_method in ['mean', 'max', 'attn'], f'{self.pool_method} is not a valid pooling method. Must be one of [`mean`,`max`,`attn`].'
        if self.pool_method in ['mean', 'max']:
            assert 'pool_dim' in kwargs, 'Pooling dimensions not given for mean or max pooling'
            self.pool_dim = kwargs['pool_dim'] #should be an int or a tuple of ints

        #INITIALIZE BASE MODEL
        self._initialize_base_model()
        self.model_name = self.get_model_name()
        self.base_config = self._base_config()
        self.base_model = self.base_model.to(device)

        # INITIALIZE CLASSIFIER (doesn't need to be overwritten)
        self.clf = Classifier(**self.clf_args)
        self.clf_config = self.clf.get_config()
        self.clf = self.clf.to(device)
        raise NotImplementedError('Classifier weight randomization with seed set')



    ### LOGGING ###
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been pretrained and finetuned
        Update for new model classes
        :return model_name: str with model name
        """
        model_name = f'BaseModel_f{self.freeze_method}_{self.pool_method}'
        if self.pt_ckpt is not None:
            model_name += '_pt'
        if self.ft_ckpt is not None:
            model_name += '_ft'
        return model_name
    
    def _base_config(self) -> Dict[str, Union[str, List[str]]]:
        """
        Get a base configuration file to append to
        :return base_config: Dict[str -> str or List of str]
        """
        base_config={'model_name':self.model_name, 'freeze_method': self.freeze_method, 'pool_method':self.pool_method,
                      'pt_ckpt': self.pt_ckpt, 'ft_ckpt': self.ft_ckpt}
        if self.pt_ckpt is not None:
            base_config['pt_ckpt'] = str(base_config['pt_ckpt'])
        if self.ft_ckpt is not None:
            base_config['ft_ckpt'] = str(base_config['ft_ckpt'])
        if self.freeze_method == 'layer':
            base_config['unfreeze_layers'] = self.unfreeze_layers
        if self.pool_method in ['mean', 'max']:
            base_config['pool_dim'] = self.pool_dim
        
        return base_config

    ### BASE MODEL INITIALIZATION ###
    @abstractmethod
    def _initialize_base_model(self):
        """
        Initialize model variable 
        Required method for model class
        """
        #SET UP BASE MODEL
        model_dict = OrderedDict()
        model_dict['linear'] =nn.Linear(1,1)
        self.base_model = nn.Sequential(model_dict) #for testing model initialization
        
        #LOAD CHECKPOINTS
        self._load_checkpoint(self.pt_ckpt)
        self._load_checkpoint(self.ft_ckpt)

        #FREEZE MODEL LAYERS
        if self.freeze_method != 'none':
            self._freeze_all()
            if self.freeze_method == 'layer':
                self._unfreeze_by_layer(self)
        
    def _load_checkpoint(self, ckpt):
        """
        Load a checkpoint for the base model from a state_dict
        Required method for model class
        Needs to be overwritten depending on how a checkpoint needs to be loaded and whether there is a difference for pretrained and finetuned loading
        
        :param ckpt: pathlike object, model checkpoint - path to state dict
        """
        if ckpt is not None:
            self.base_model.load_state_dict(torch.load(ckpt, weights_only=True))
        
    def _freeze_all(self):
        """
        Freeze the entire model
        Will not need to be overwritten
        """
        for param in self.base_model.parameters():
            param.requires_grad = False 

    @abstractmethod
    def _unfreeze_by_layer(self):
        """
        Unfreeze specific model layers
        Needs to be overwritten depending on how each model stores layers
        """
        assert all([isinstance(v, str) for v in self.unfreeze_layers]), f'Freeze layers should be given as a string layer name.'
        for name, layer in self.base_model.named_children():
            if name in self.unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True

    ### POOLING ###
    def pooling(self, x:torch.Tensor) -> torch.Tensor:
        """
        Mean/max pooling
        May need to change dimensions depending on model
        :param x: torch tensor, input
        """
        if self.pool_method == 'mean':
            return torch.mean(x, self.pool_dim)
        elif self.pool_method == 'max': 
            return torch.max(x, self.pool_dim)
        else:
            raise NotImplementedError(f'{self.pool_method} not yet implemented.')

    ### FORWARD METHOD ###
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Run model
        May need to be overwritten depending on model

        :param x: torch tensor, model input
        :return: torch tensor, model output
        """
        base_x = self.base_model(x)
        pool_x = self.pooling(base_x)
        return self.clf(pool_x)

    ### SAVING ###
    def save_config(self, config:Dict,):
        """
        Save a config dictionary
        :param config: Dict
        """
        save_path = self.out_dir / 'config.json'
        with open(str(save_path), "w") as outfile:
            json.dump(config, outfile)


    def save_model(self):
        """
        Save the model components
        """
        bm_path = self.out_dir / self.model_name+'.pt'
        torch.save(self.base_model.state_dict(), bm_path)

        clf_path = self.out_dir / self.clf_config['clf_name']+'.pt'
        torch.save(self.clf.state_dict(), clf_path)