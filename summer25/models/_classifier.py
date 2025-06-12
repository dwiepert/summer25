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
    :param seed: int, random seed (default = 42)

    """
    def __init__(self, in_features:int, out_features:int, nlayers:int=2, 
                 activation:str='sigmoid', ckpt:Optional[Union[str,Path]]=None, seed:int=42):
        
        super(Classifier, self).__init__()
        #INITIALIZE VARIABLES
        self.in_feats = in_features
        self.out_feats = out_features
        self.nlayers = nlayers
        self.activation = activation 
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
        model_dict = OrderedDict()
        for i in range(self.nlayers):
            model_dict[f'linear{i+1}'] = nn.Linear(self.params['in_feats'][i], self.params['out_feats'][i])
            if i+1 != self.nlayers:
                model_dict[f'{self.activation}{i+1}'] = self._get_activation_layer()
        self.classifier = nn.Sequential(model_dict)
        
        #https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        print('TODO: random initialization of weights')
        #self.classifier.apply(self.classifier._init_weights)
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
            self.params['in_feats'] = [self.in_feats,self.in_feats]
            self.params['out_feats'] = [self.in_feats,self.out_feats]
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

    def _load_checkpoint(self):
        """
        Load a checkpoint for the model from a state_dict
        
        :param ckpt: pathlike object, model checkpoint - path to state dict
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
    
    def save_classifier(self, out_dir:Union[Path, str]):
        """
        Save the model components
        :param out_dir: pathlike, location to save model to
        """
        if not isinstance(out_dir, Path): out_dir=Path(out_dir)
        clf_path = out_dir / 'weights' 
        clf_path.mkdir(exist_ok=True)
        clf_path = clf_path / (self.config['clf_name']+'.pt')
        if clf_path.exists(): print(f'Overwriting existing classifier head at {str(clf_path)}')
        torch.save(self.classifier.state_dict(), clf_path)