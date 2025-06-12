"""
Model class for hugging face models

Author(s): Daniela Wiepert
Last modified: 06/2025
"""

#IMPORT
##built-in
import copy
import os
from pathlib import Path
import random
import shutil
from typing import List, Union, Dict, Optional, Tuple

##third-party
from huggingface_hub import snapshot_download
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, WhisperModel

##local
from ._base_model import BaseModel
from ._classifier import Classifier
from summer25.constants import _MODELS

class HFModel(BaseModel):
    """
    Model class for hugging face models 

    :param out_dir: Pathlike, output directory to save to
    :param model_type: str, hugging face model type for naming purposes
    :param keep_extractor: bool, keep base extractor frozen (default=True)
    :param freeze_method: str, freeze method for base pretrained model (default=all)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param ft_ckpt: pathlike, path to finetuned base model checkpoint (default=None)
    :param out_features: int, number of output features from classifier (number of classes) (default = 1)
    :param nlayers: int, number of layers in classification head (default = 2)
    :param activation: str, activation function to use in classification head (default = 'sigmoid')
    :param seed: int, specify random seed for ensuring reproducibility
    :param device: torch device
    :param kwargs: additional arguments for optional parameters (e.g., unfreeze_layers if freeze_method is layer, delete_download if you want to remove local versions of checkpoints after downloading)
    """
    def __init__(self, 
                 out_dir:Union[Path, str], model_type:str, keep_extractor:bool=True,
                 freeze_method:str = 'all', pool_method:str = 'mean', ft_ckpt:Optional[Union[Path,str]]=None, 
                 out_features:int=1, nlayers:int=2, activation:str='sigmoid',
                 seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs):

        self.out_dir = out_dir 
        if not isinstance(self.out_dir, Path): self.out_dir = Path(self.out_dir)

        ## BASE MODEL ARGS
        self.model_type = model_type 
        self.hf_hub = _MODELS[self.model_type]['hf_hub']
        if "delete_download" in kwargs:
            self.delete_download = kwargs.pop("delete_download")
        else:
            self.delete_download = False
        #TODO: try this and see if it works
        self.keep_extractor = keep_extractor
        self.use_featext = _MODELS[self.model_type]['use_featext']

        ## CLF ARGS
        self.in_features = _MODELS[self.model_type]['in_features']
        self.out_features = out_features
        self.nlayers = nlayers
        self.activation = activation

        #TODO: confirm weights are initialized properly - NEED TO ALSO DO THIS FOR THE CLASSIFIER!!!!!!!!!!

  
        super().__init__(out_dir=out_dir, model_type=self.model_type,
                         freeze_method=freeze_method, pool_method=pool_method, ft_ckpt=ft_ckpt,
                         in_features=self.in_features, out_features=self.out_features, nlayers=self.nlayers, activation=self.activation, 
                         device=device, seed=seed, **kwargs)
    
        
        if self.local_path and not self.delete_download:
            self.base_config['pt_ckpt'] = self.local_path

        self.config = {'model_type':self.model_type,'seed':self.seed, 'keep_extractor': self.keep_extractor,
                       'clf_config': self.clf_config, 'base_config':self.base_config}
        
        self.save_config()
 
    
    def get_model_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been finetuned
        Update for new model classes
        :return model_name: str with model name
        """
        model_name = f'{self.model_type}_seed{self.seed}_f{self.freeze_method}_{self.pool_method}'
        if self.keep_extractor:
            model_name += '_keepext'
        if self.ft_ckpt is not None:
            model_name += '_ft'
        return model_name

    def _load_model(self):
        if self.pt_ckpt:
            if self.use_featext:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.pt_ckpt)
            self.base_model = AutoModel.from_pretrained(self.pt_ckpt, output_hidden_states=True)
        else:    
            try:
                if self.use_featext:
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hf_hub, trust_remote_code=True)
                self.base_model = AutoModel.from_pretrained(self.hf_hub, output_hidden_states=True, trust_remote_code=True)
                self.local_path=None
            except Exception as e:
                print('Downloading model locally...')
                self.local_path = Path(f'./{self.hf_hub}').absolute()
                self.local_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(repo_id=self.hf_hub, local_dir=str(self.local_path))
                if self.use_featext:
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hf_hub)
                self.base_model = AutoModel.from_pretrained(self.local_path, output_hidden_states=True)
                if self.delete_download:
                    print('Deleting local copy of checkpoint')
                    shutil.rmtree(str(self.local_path))

                    bp = Path('.').absolute()
                    curr_parent = self.local_path.parent
                    while bp.name != curr_parent.name: 
                        os.rmdir(curr_parent)
                        temp = curr_parent.parent 
                        curr_parent = temp
                   
    def _initialize_base_model(self):
        """
        Initialize base hugging face models
        """
        print(f'Loading model {self.model_type} from Hugging Face Hub...')

        self._load_model()      
        self.is_whisper_model = isinstance(self.base_model, WhisperModel)
        
        if self.keep_extractor and not self.is_whisper_model:
            ext_state_dict = copy.deepcopy(self.base_model.feature_extractor.state_dict())
        
        print('TODO: check that weights are initialized properly? Is _init_weights good for Whisper too???')
        self.base_model.apply(self.base_model._init_weights)

        if self.keep_extractor and not self.is_whisper_model:
            self.base_model.feature_extractor.load_state_dict(ext_state_dict)
            del ext_state_dict

        self._load_checkpoint(self.ft_ckpt)

        #FREEZE MODEL LAYERS
        if self.freeze_method != 'none':
            self._freeze_all()
            if self.freeze_method == 'layer':
                self._unfreeze_by_layer()

        if self.use_featext:
            self.feature_extractor_kwargs = {}
            if self.is_whisper_model:
                self.features_key = 'input_features'
                self.feature_extractor_kwargs['return_attention_mask'] = True
            else:
                self.features_key = 'input_values'
    
    def forward(self, sample:Dict):
        """
        Overwritten forward loop. 

        :param sample: batched sample from a DataLoader object that minimally contains a `waveform` key storing the tensor of the loaded audio
        :return: classifier output
        """
        print('Where to use processor??? here ok??? or needs to be outside?')
        if self.use_featext:
            preprocessed_wav = self.feature_extractor(list(sample['waveform']).cpu().numpy(),
                                                        return_tensors='pt', 
                                                        sampling_rate = self.feature_extractor.sampling_rate,
                                                        **self.feature_extractor_kwargs)
        
        if self.is_whisper_model:
            output = self.base_model.encoder(preprocessed_wav[self.features_key].to(self.device))
        else:
            output = self.base_model(preprocessed_wav[self.features_key].to(self.device))

        pooled = self.pooling(output)
 
        return self.clf(pooled)
