"""
Classifier class for training

Author(s): Daniela Wiepert
Last modified: 07/2025
"""

#IMPORT
##built-in
from collections import OrderedDict
import math
import os
from pathlib import Path
from typing import Union, Dict, Optional

##third-party
import torch
import torch.nn as nn

#local
from summer25.io import upload_to_gcs, download_to_local, search_gcs

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input sequence.
    From "Attention is All You Need" (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Tensor with added positional information.
        """
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class TransformerClassifier(nn.Module):
    """
    Transformer classifier class

    :param in_features: integer, input size
    :param out_features: integer, number of classes
    :param nlayers: integer, number of classifier layers (default=2)
    :param dropout: float, dropout level (default = 0.0)
    :param num_heads:int, number of encoder heads in using transformer build (default = 4)
    :param seed: int, random seed (default = 42)
    """
    def __init__(self, in_features:int, out_features:int, nlayers:int=2,
                 dropout:float=0.0, num_heads:int=4, seed:int=42):
        self.in_feats = in_features
        self.out_feats = out_features
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_heads
        self.seed = seed
        torch.manual_seed(self.seed)

        self._build_classifier()
      
    
    def _build_classifier(self):
        """
        Build transformer classifier
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.in_feats,
            nhead=self.num_heads,
            dim_feedforward=self.in_feats,
            dropout=self.dropout,
            batch_first=True,
            activation='relu'
        )

        self.pos_encoder = PositionalEncoding(self.in_feats, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.nlayers
        )
        
        self.fc = nn.Linear(self.in_feats, self.out_feats)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Classifier forward function
        :param x: torch.Tensor, input
        :return: torch tensor, classifier output
        """
        x = x * math.sqrt(self.in_feats)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(src)
        pooled_output = output.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits
           

class LinearClassifier(nn.Module):
    """
    Linear classifier class

    :param in_features: integer, input size
    :param out_features: integer, number of classes
    :param nlayers: integer, number of classifier layers (default=2)
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param activation: str, activation function to use (default=relu)
    :param binary:bool, specify whether output is making binary decisions (default=True)
    :param seed: int, random seed (default = 42)
    """
    def __init__(self, in_features:int, out_features:int, nlayers:int=2, bottleneck:int=None, layernorm:bool=False, 
                 dropout:float=0.0, activation:str='relu', binary:bool=True, seed:int=42):
        self.in_feats = in_features
        self.out_feats = out_features
        self.bottleneck = bottleneck
        self.nlayers = nlayers

        if self.nlayers == 2 and not self.bottleneck:
            self.bottleneck = self.in_feats 

        self.activation = activation 
        self.binary = binary
        
        self.seed = seed
        torch.manual_seed(self.seed)

        self._build_classifier()

    def _params(self):
        """
        Get linear classifier parameters based on input parameters
        """
        self.params = {}
        if self.nlayers == 1:
            self.params['in_feats'] = [self.in_feats]
            self.params['out_feats'] = [self.out_feats]
        elif self.nlayers == 2:
            self.params['in_feats'] = [self.in_feats,self.bottleneck]
            self.params['out_feats'] = [self.bottleneck, self.out_feats]
        else:
            raise NotImplementedError(f'Classifier parameters not yet implemented for given inputs.')

        def _get_activation_layer(self, activation) -> nn.Module:
        """
        Create an activation layer based on specified activation function

        :return: nn.Module activation layer
        """
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'relu':
            return nn.ReLU()
        else:
            raise NotImplementedError(f'{activation} not yet implemented.')

    def _build_classifier(self) -> Dict[str, nn.Module]:
        """
        Build classifier ordered dictionary

        :return model dict: Dict[nn.Module], ordered dictionary with model layers
        """
        self._params()
        model_dict = OrderedDict()
        for i in range(self.nlayers):
            model_dict[f'linear{i}'] = nn.Linear(self.params['in_feats'][i], self.params['out_feats'][i])
            if i+1 != self.nlayers:
                if self.layernorm:
                    model_dict[f'layernorm{i}'] = nn.LayerNorm(self.params['out_feats'][i])
                
                model_dict[f'{self.activation}{i}'] = self._get_activation_layer(self.activation)
                if self.dropout != 0.0:
                    model_dict[f'dropout{i}'] = nn.Dropout(self.dropout)
        if self.binary:
            model_dict['sigmoid'] = self._get_activation_layer('sigmoid')
        
        self.classifier = nn.Sequential(model_dict)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Classifier forward function
        :param x: torch.Tensor, input
        :return: torch tensor, classifier output
        """
        return self.classifier(x)
        
class Classifier(nn.Module):
    """
    Classifier class with flexible variables for initializing a variety of classifier configurations
    Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    
    :param in_features: integer, input size
    :param out_features: integer, number of classes
    :param nlayers: integer, number of classifier layers (default=2)
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param activation: str, activation function to use (default=relu)
    :param binary:bool, specify whether output is making binary decisions (default=True)
    :param separate:bool, true if each feature gets a separate classifier head
    :param clf_type:str, specify layer type ['linear','transformer'] (default='linear')
    :param num_heads:int, number of encoder heads in using transformer build (default = 4)
    :param seed: int, random seed (default = 42)
    """
    def __init__(self, in_features:int, out_features:int, nlayers:int=2, bottleneck:int=None, layernorm:bool=False, 
                 dropout:float=0.0, activation:str='relu', binary:bool=True, separate:bool=True,
                 clf_type:str='linear', num_heads:int=4, seed:int=42):
        
        super(Classifier, self).__init__()
        #INITIALIZE VARIABLES
        self.in_feats = in_features
        self.out_feats = out_features
        self.bottleneck = bottleneck
        self.nlayers = nlayers
        self.separate = separate

        if self.nlayers == 2 and not self.bottleneck:
            self.bottleneck = self.in_feats 

        self.activation = activation 
        self.binary = binary
        self.clf_type = clf_type
        self.num_heads = num_heads
        self.layernorm = layernorm 
        self.dropout = dropout

        #SET SEED
        self.seed = seed
        torch.manual_seed(self.seed)

        #ASSERTIONS
        #SET UP CLASSIFIER
        self._params()
        if self.clf_type == 'linear':
            model_dict = self._build_linear_classifier()
        elif self.clf_type == 'transformer':
            model_dict = self._build_transformer_classifier()
        else:
            raise NotImplementedError
        self.classifier = nn.Sequential(model_dict)
        self.classifier.apply(self._init_weights)

        #SET UP CLF CONFIG
        clf_name = self.get_clf_name()
        self.config = {'clf_name':clf_name, 'in_features':self.in_feats, 'out_features':self.out_feats, 
                       'nlayers':self.nlayers, 'activation':self.activation, 'binary': self.binary, 'clf_type':self.clf_type, 
                       'num_heads': self.num_heads, 'seed': self.seed, 'bottleneck': self.bottleneck, 'layernorm':self.layernorm,
                       'dropout':self.dropout}
        
    def _get_classifiers(self): 
        """
        Build classifier based on arguments
        """
        if self.separate: 
            n_classifers = self.out_feats
            out_feats = 1
        else:
            n_classifiers = 1
            out_feats = self.out_feats
        
        self.classifiers = []
        for n in n_classifiers:
            if self.clf_type == 'linear':
                self.classifiers.append(LinearClassifier(in_features=self.in_feats, out_features=out_feats, nlayers=self.nlayers, 
                                                         bottleneck=self.bottleneck, layernorm=self.layernorm, dropout = self.dropout, activation=self.activation,
                                                         binary=self.binary, seed=self.seed).apply(self._init_weights))
            elif self.clf_type == 'transformer':
                self.classifiers.append(TransformerClassifier(in_features=self.in_feats, out_features=out_feats, nlayers=self.nlayers,
                                                              dropout=self.dropout, num_heads = self.num_heads, seed=self.seed).apply(self._init_weights))
            else:
                raise NotImplementedError(f'{self.clf_type} not an implemented classifier')
        
        self.classifiers = nn.ModuleList(self.classifiers)

    def _init_weights(self, layer:nn.Module):
        """
        Randomize classifier weights

        :param layer: nn.Module, model layer
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight) #EXPLAIN WHY KAIMING
 
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Classifier forward function
        :param x: torch.Tensor, input
        :return: torch tensor, classifier output
        """
        preds = []
        for clf in self.classifiers:
            pred = clf(x)
            preds.append(pred)

        logits = torch.column_stack(preds)
        return logits

    def get_config(self) -> Dict[str, Union[str, int]]:
        """
        Return classifier config
        :return: configuration dictionary for classifier
        """
        return self.config
    
    def get_clf_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been pretrained and finetuned
        Update for new model classes
        :return model_name: str with classifier model name
        """
        model_name = f'Classifier_{self.clf_type}_in{self.in_feats}_out{self.out_feats}_{self.activation}_n{self.nlayers}'
        if self.ckpt is not None:
            model_name += '_ct'
        return model_name
    