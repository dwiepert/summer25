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
import os
from pathlib import Path
from typing import List, Union, Dict, Optional

##third-party
import torch
import torch.nn as nn

##local
from summer25.constants import _MODELS, _FREEZE, _POOL, _FINETUNE
from ._attention_pooling import SelfAttentionPooling

class BaseModel(nn.Module):
    """
    Base model class
    Includes all shared parameters

    :param model_type: str, type of model being initialized
    :param out_dir: pathlike, path to directory for saving all model information
    :param finetune_method: str, specify finetune method (default=None)
    :param freeze_method: str, freeze method for base pretrained model (default=required-only)
    :param unfreeze_layers: List[str], optionally give list of layers to unfreeze (default = None)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param pt_ckpt: pathlike, path to base pretrained model checkpoint (default=None)
    :param ft_ckpt: pathlike, path to finetuned base model checkpoint (default=None)
    :param clf_ckpt: pathlike, path to finetuned classifier checkpoint (default = None)
    :param in_features: int, number of input features to classifier (based on output layers of the base model) (default = 768)
    :param out_features: int, number of output features (number of classes) (default = 1)
    :param nlayers: int, number of layers in classification head (default = 2)
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param binary:bool, specify whether output is making binary decisions (default=True)
    :param activation: str, activation function to use for classification head (default=relu)
    :param seed: int, random seed (default = 42)
    :param device: torch device (default = cuda)
    """
    def __init__(self, model_type:str, out_dir:Union[Path, str], finetune_method:str='none', 
                 freeze_method:str = 'required-only', unfreeze_layers:Optional[List[str]]=None, 
                 pool_method:str = 'mean', pool_dim:Optional[Union[int, tuple]] = None,
                 pt_ckpt:Optional[Union[Path, str]]=None, ft_ckpt:Optional[Union[Path,str]]=None, clf_ckpt:Optional[Union[Path,str]]=None,  
                 in_features:int=768, out_features:int=1, nlayers:int=2, bottleneck:int=None, layernorm:bool=False, dropout:float=0.0, binary:bool=True,
                 activation:str='relu', seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(BaseModel, self).__init__()

        # INITIALIZE VARIABLES
        self.model_type = model_type
        self.out_dir = out_dir
        if not isinstance(self.out_dir, Path): self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pt_ckpt=pt_ckpt
        self.ft_ckpt=ft_ckpt
        self.clf_ckpt = clf_ckpt
        self.finetune_method = finetune_method
        self.freeze_method = freeze_method
        self.pool_method = pool_method
        self.seed = seed
        self.device = device

        #SET SEED
        torch.manual_seed(self.seed)

        self.clf_args = {'in_features':in_features, 'out_features':out_features, 'nlayers':nlayers,
                        'activation':activation, 'bottleneck':bottleneck, 'layernorm':layernorm, 'binary':binary,
                        'dropout':dropout, 'seed': self.seed, 'ckpt': self.clf_ckpt}
        
        # ASSERTIONS
        assert self.model_type in list(_MODELS.keys()), f'{self.model_type} is an invalid model type. Choose one of {list(_MODELS.keys())}.'
        if self.pt_ckpt is not None:
            if not isinstance(self.pt_ckpt, Path): self.pt_ckpt = Path(self.pt_ckpt)
            assert self.pt_ckpt.exists(), f'Pretrained model path {self.pt_ckpt} does not exist.'
        if self.ft_ckpt is not None:
            if not isinstance(self.ft_ckpt, Path): self.ft_ckpt = Path(self.ft_ckpt)
            assert self.ft_ckpt.exists(), f'Finetuned model path {self.ft_ckpt} does not exist.'
            if self.ft_ckpt.is_dir(): 
                assert os.listdir(self.ft_ckpt) != [], 'Finetuned checkpoint must be a non-empty directory'
            else: 
                assert '.pt' in str(self.ft_ckpt) or '.pth' in str(self.ft_ckpt), 'Must give .pt or .pth if not giving a directory'
        ### finetune method 
        assert self.finetune_method in _FINETUNE, f'self.finetune_method is not a valid finetuning method. Choose one of {_FINETUNE}.'
        ### freeze method
        assert self.freeze_method in _FREEZE, f'{self.freeze_method} is not a valid freeze method. Choose one of {_FREEZE}.'
        self.unfreeze_layers = unfreeze_layers
        if self.freeze_method == 'layer':
            assert unfreeze_layers is not None, 'Layers to unfreeze not given as input'
            assert isinstance(self.unfreeze_layers, list), 'Unfreeze layers expects a list.'
            assert all([isinstance(l,str) for l in self.unfreeze_layers]) or all([isinstance(l,int) for l in self.unfreeze_layers]), 'Unfreeze layers expects a list of str or list of ints.'

        assert self.pool_method in _POOL, f'{self.pool_method} is not a valid pooling method. Choose one of {_POOL}.'
        if pool_dim is not None:
            self.pool_dim = pool_dim
        elif 'pool_dim' in _MODELS[model_type]:
            self.pool_dim = _MODELS[model_type]['pool_dim']
        else:
            raise ValueError('Pool dim not given.')
        
        if self.pool_method in ['mean', 'max']:
            assert isinstance(self.pool_dim, int) or (isinstance(self.pool_dim, tuple)), 'Pooling dimensions must be single integer or tuple of integers'
            if isinstance(self.pool_dim, tuple):
                assert all(isinstance(i,int) for i in self.pool_dim), 'All pooling dimensions in a tuple must be integers.'
        else:
            assert isinstance(self.pool_dim, int) and self.pool_dim == 1, 'Self attention pooling only works with a pooling dimension of 1.'
            self.attention_pooling = SelfAttentionPooling(input_dim=in_features)
        
        self.base_config = self._base_config()

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
        base_config={'freeze_method': self.freeze_method, 'pool_method':self.pool_method}
        if self.pt_ckpt is not None:
            base_config['pt_ckpt'] = str(self.pt_ckpt)
        if self.ft_ckpt is not None:
            base_config['ft_ckpt'] = str(self.ft_ckpt)
        if self.freeze_method == 'layer':
            base_config['unfreeze_layers'] = self.unfreeze_layers
        
        return base_config

    ### BASE MODEL INITIALIZATION ###
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
            try:
                self.base_model.load_state_dict(torch.load(ckpt, weights_only=True))
            except:
                raise ValueError('Finetuned checkpoint could not be loaded. Weights may not be compatible with the initialized models.')
    
    def _freeze_all(self):
        """
        Freeze the entire model
        Will not need to be overwritten
        """
        for param in self.base_model.parameters():
            param.requires_grad = False 

    def _unfreeze_by_layer(self, unfreeze_layers):
        """
        Unfreeze specific model layers
        May need to be overwritten depending on how each model stores layers
        """
        assert all([isinstance(v, str) for v in unfreeze_layers]), f'Freeze layers should be given as a string layer name.'
        for name, param in self.base_model.named_parameters():
            if any([u in name for u in unfreeze_layers]):
                param.requires_grad=True

    ### POOLING ###
    def pooling(self, x:torch.Tensor, attn_mask:torch.Tensor=None) -> torch.Tensor:
        """
        Mean/max pooling
        May need to change dimensions depending on model
        :param x: torch tensor, input
        """
        if self.pool_method == 'mean':
            if attn_mask is not None:
                return x.sum(dim=self.pool_dim) / attn_mask.sum(dim=self.pool_dim).view(-1, 1)
            else:
                return torch.mean(x, self.pool_dim)
        elif self.pool_method == 'max': 
            return torch.max(x, self.pool_dim).values
        else:
            if attn_mask is not None:
                lengths = torch.count_nonzero(attn_mask, dim=1)
            else:
                lengths = torch.count_nonzero(x[:,:,0], dim=1)
            return self.attention_pooling(x, lengths)
    
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
    def save_config(self):
        """
        Save a config dictionary
        :param config: Dict
        """
        save_path = self.out_dir / 'configs'
        save_path.mkdir(exist_ok=True)
        save_path = save_path / 'model_config.json'
        if save_path.exists(): print(f'Overwriting model config file saved at {str(save_path)}')

        with open(str(save_path), "w") as outfile:
            json.dump(self.config, outfile)


    def save_base_model(self, name:str):
        """
        Save the model components
        :param name: str, save name for model
        """
        bm_path = self.out_dir / 'weights'

        bm_path.mkdir(exist_ok=True)
        bm_path = bm_path / (name+'.pt')

        if bm_path.exists(): print(f'Overwriting existing model at {str(bm_path)}')
        torch.save(self.base_model.state_dict(), bm_path)

    