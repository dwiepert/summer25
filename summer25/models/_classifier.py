"""
Classifier class for training

Author(s): Daniela Wiepert
Last modified: 06/2025
"""

#IMPORT
##built-in
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict, Optional

##third-party
import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    Classifier class with flexible variables for initializing a variety of classifier configurations

    :param in_features: integer, input size
    :param out_features: integer, number of classes
    :param nlayers: integer, number of classifier layers (default=2)
    :param activation: str, activation function to use (default=sigmoid)
    :param ckpt: pathlike, optional path to trained classifier (default=none)

    """
    def __init__(self, in_features:int, out_features:int, nlayers:int=2, 
                 activation:str='sigmoid', ckpt:Optional[Union[str,Path]]=None):
        #INITIALIZE VARIABLES
        self.in_feats = in_features
        self.out_feats = out_features
        self.nlayers = nlayers
        self.activation = activation 
        self.ckpt = ckpt

        if not isinstance(self.ckpt, Path): self.ckpt = Path(self.ckpt)

        #ASSERTIONS
        assert self.activation in ['sigmoid'], f'{self.activation} is not a valid activation function.'
        max_layers = 1
        assert self.nlayers <= max_layers, f'Classifier class cannot handle {self.nlayers}. Must be less than {max_layers}.'
        #TODO: assertions for in and out feats

        #SET UP CLASSIFIER
        self._params()
        model_dict = OrderedDict()
        for i in range(self.nlayers):
            model_dict[f'linear{i+1}'] = nn.Linear(self.params['in_feats'][i], self.params['out_feats'][i])
            if i+1 != self.nlayers:
                model_dict[f'{self.activation}{i+1}'] = self._get_activation_layer()
        self.classifier = nn.Sequential(model_dict)
        self._load_checkpoint()

        #SET UP CLF CONFIG
        clf_name = self.get_clf_name()
        self.config = {'clf_name':clf_name, 'in_features':self.in_feats, 'out_features':self.out_feats, 
                       'nlayers':self.nlayers, 'activation':self.activation}
        if self.ckpt is not None:
            self.config['ckpt'] = str(self.config['ckpt']) 

    def _params(self):
        """
        Get classifier parameters based on input parameters
        """
        self.params = {}
        if self.nlayers == 2 and self.in_feats == 1 and self.out_feats == 1:
            self.params['in_feats'] = [1,1]
            self.params['out_feats'] = [1,1]
        else:
            raise NotImplementedError(f'Classifier parameters not yet implemented for given inputs.')
        
    def _get_activation_layer(self) -> nn.Module:
        """
        Create an activation layer based on specified activation function

        :return: nn.Module activation layer
        """
        if self.activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise NotImplementedError(f'{self.activation} not yet implemented.')
            #TODO: add

    def _load_checkpoint(self, ckpt):
        """
        Load a checkpoint for the model from a state_dict
        
        :param ckpt: pathlike object, model checkpoint - path to state dict
        """
        if ckpt is not None:
            self.classifier.load_state_dict(torch.load(ckpt, weights_only=True))
    
    def forward(self, x) -> torch.Tensor:
        """
        Classifier forward function
        :return: torch tensor, classifier output
        """
        return self.classifier(x)
    
    def get_config(self) -> Dict[str, Union[str, int]]:
        """
        Return classifier config
        """
        return self.config
    
    def get_clf_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been pretrained and finetuned
        Update for new model classes
        :return model_name: str with model name
        """
        model_name = f'Classifier_in{self.in_feats}_out{self.out_feats}_{self.activation}_n{self.nlayers}'
        if self.ckpt is not None:
            model_name += '_ct'
        return model_name