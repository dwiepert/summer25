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
    Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    
    :param in_features: integer, input size
    :param out_features: integer, number of classes
    :param nlayers: integer, number of classifier layers (default=2)
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param activation: str, activation function to use (default=relu)
    :param ckpt: pathlike, optional path to trained classifier (default=none)
    :param seed: int, random seed (default = 42)

    """
    def __init__(self, in_features:int, out_features:int, nlayers:int=2, bottleneck:int=None, layernorm:bool=False, 
                 dropout:float=0.0, activation:str='relu', ckpt:Optional[Union[str,Path]]=None, seed:int=42):
        
        super(Classifier, self).__init__()
        #INITIALIZE VARIABLES
        self.in_feats = in_features
        self.out_feats = out_features
        self.bottleneck = bottleneck
        self.nlayers = nlayers

        if self.nlayers == 2 and not self.bottleneck:
            self.bottleneck = self.in_feats 

        self.activation = activation 
        self.layernorm = layernorm 
        self.dropout = dropout
        self.ckpt = ckpt

        #SET SEED
        self.seed = seed
        torch.manual_seed(self.seed)

        if self.ckpt:
            if not isinstance(self.ckpt, Path): self.ckpt = Path(self.ckpt)
            assert self.ckpt.exists(), f'Cannot load from given checkpoint: {self.ckpt}'
            if self.ckpt.suffix != '.pt' and self.ckpt.suffix != '.pth':
                poss_files = [p for p in self.ckpt.rglob("*.pt")]
                poss_files += [p for p in self.ckpt.rglob("*.pth")]

                assert poss_files != [], 'No checkpoints exist in given directory.'
                paths = [str(p) for p in poss_files]
                paths2 = [p for p in paths if 'Classifier' in p]
                self.ckpt = paths2[0]

        #ASSERTIONS
        #SET UP CLASSIFIER
        self._params()
        model_dict = self._build_classifier()
        self.classifier = nn.Sequential(model_dict)
        self.classifier.apply(self._init_weights)
        self._load_checkpoint()

        #SET UP CLF CONFIG
        clf_name = self.get_clf_name()
        self.config = {'clf_name':clf_name, 'in_features':self.in_feats, 'out_features':self.out_feats, 
                       'nlayers':self.nlayers, 'activation':self.activation}
        
        if self.ckpt is not None:
            self.config['clf_ckpt'] = str(self.ckpt) 

    def _params(self):
        """
        Get classifier parameters based on input parameters
        """
        self.params = {}
        if self.nlayers == 1:
            self.params['in_feats'] = [self.in_feats]
            self.params['out_feats'] = [self.out_feats]
        elif self.nlayers == 2:
            self.params['in_feats'] = [self.in_feats,self.bottleneck]
            self.params['out_feats'] = [self.bottleneck,self.out_feats]
        else:
            raise NotImplementedError(f'Classifier parameters not yet implemented for given inputs.')
        
    def _get_activation_layer(self) -> nn.Module:
        """
        Create an activation layer based on specified activation function

        :return: nn.Module activation layer
        """
        if self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'relu':
            return nn.ReLU()
        else:
            raise NotImplementedError(f'{self.activation} not yet implemented.')

    def _build_classifier(self) -> Dict[str, nn.Module]:
        """
        Build classifier ordered dictionary

        :return model dict: Dict[nn.Module], ordered dictionary with model layers
        """
        model_dict = OrderedDict()
        for i in range(self.nlayers):
            model_dict[f'linear{i}'] = nn.Linear(self.params['in_feats'][i], self.params['out_feats'][i])
            if i+1 != self.nlayers:
                if self.layernorm:
                    model_dict[f'layernorm{i}'] = nn.LayerNorm(self.params['out_feats'][i])
                
                model_dict[f'{self.activation}{i}'] = self._get_activation_layer()
                if self.dropout != 0.0:
                    model_dict[f'dropout{i}'] = nn.Dropout(self.dropout)
        return model_dict 
    
    def _init_weights(self, layer:nn.Module):
        """
        Randomize classifier weights

        :param layer: nn.Module, model layer
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform(layer.weight) #EXPLAIN WHY KAIMING

    def _load_checkpoint(self):
        """
        Load a checkpoint for the model from a state_dict
        """
        if self.ckpt is not None:
            try:
                self.classifier.load_state_dict(torch.load(self.ckpt, weights_only=True))
            except:
                raise ValueError('Classifier checkpoint could not be loaded. Weights may not be compatible with the initialized models.')
                
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Classifier forward function
        :return: torch tensor, classifier output
        """
        return self.classifier(x)
    
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
        model_name = f'Classifier_in{self.in_feats}_out{self.out_feats}_{self.activation}_n{self.nlayers}'
        if self.ckpt is not None:
            model_name += '_ct'
        return model_name
    
    def save_classifier(self, name:str, out_dir:Union[Path, str]):
        """
        Save the model components
        :param name: str, name to save classifier to
        :param out_dir: pathlike, location to save model to
        """
        if not isinstance(out_dir, Path): out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok = True)
        clf_path = out_dir / (name+'.pt')
        if clf_path.exists(): print(f'Overwriting existing classifier head at {str(clf_path)}')
        torch.save(self.classifier.state_dict(), clf_path)