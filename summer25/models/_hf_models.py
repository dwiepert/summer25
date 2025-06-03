"""
Model class for hugging face models

Author(s): Daniela Wiepert
Last modified: 06/2025
"""

#IMPORT
##built-in
import copy
from pathlib import Path
import random
from typing import List, Union, Dict, Optional, Tuple

##third-party
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, WhisperModel

##local
from ._base_model import BaseModel

class HFModel(BaseModel):
    """
    Model class for hugging face models 

    :param model_type: str
    :param hf_path: str, either string name for transformers folder or path to local directory
    :param freeze_method: str, freeze method for base pretrained model (default=all)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param ft_ckpt: pathlike, path to finetuned base model checkpoint (default=None)
    :param kwargs: additional arguments for optional parameters (e.g., pool_dim for mean/max pooling and unfreeze_layers if freeze_method is layer)
    """
    def __init__(self, model_type:str, hf_path:str, use_featext:bool, target_sample_rate:int, out_dir:Union[Path, str], freeze_extractor:bool=True, freeze_method:str = 'all', pool_method:str = 'mean',
                 ft_ckpt:Optional[Union[Path,str]]=None, seed:int=42, **kwargs):
        
        self.model_type = model_type 
        self.hf_path = hf_path
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        #TODO: confirm weights are initialized properly - NEED TO ALSO DO THIS FOR THE CLASSIFIER!!!!!!!!!!

        self.freeze_extractor = freeze_extractor
        self.use_featext = use_featext
        self.target_sample_rate = target_sample_rate

        super().__init__(out_dir=out_dir, clf_args=self._initialize_clf_args(), freeze_method=freeze_method, pool_method=pool_method, ft_ckpt=ft_ckpt, **kwargs)

        self.model_name = self.get_model_name()
        self.config = {'model_name': self.model_name, 'hf_path':self.hf_path, 'seed':self.seed,
                       'freeze_extractor': self.freeze_extractor, 'use_featext': self.use_featext, 'target_sample_rate': self.target_sample_rate,
                       'clf_config': self.clf_config, 'base_config':self.base_config}
        self.save_config()

    def _initialize_clf_args(self):
        raise NotImplementedError('Have to figure out clf arguments')
        pass
    
    def get_model_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been finetuned
        Update for new model classes
        :return model_name: str with model name
        """
        model_name = f'{self.model_type}_seed{self.seed}_f{self.freeze_method}_{self.pool_method}'
        if self.use_featext:
            model_name += '_featext'
        if self.freeze_extractor:
            model_name += '_freezeext'
        if self.ft_ckpt is not None:
            model_name += '_ft'
        return model_name

    def _initialize_base_model(self):
        """
        Initialize base hugging face models
        """
        print(f'Loading model {self.model_type} from Hugging Face Hub...')

        if self.use_featext:
            #check when to use separate feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hf_path)
            self.target_sample_rate = self.feature_extractor.sampling_rate
        else:
            self.feature_extractor = None
    
        self.base_model = AutoModel.from_pretrained(self.hf_path, output_hidden_states=True, trust_remote_code=True)
        self.is_whisper_model = isinstance(self.base_model, WhisperModel)
        if self.freeze_extractor:
            #TODO: check when to freeze extractor
            ext_state_dict = copy.deepcopy(self.base_model.feature_extractor.state_dict())
        
        self.base_model.apply(self.base_model._init_weights)

        if self.freeze_extractor:
            self.base_model.feature_extractor.load_state_dict(ext_state_dict)
            del ext_state_dict

        self._load_checkpoint(self.ft_ckpt)

        #FREEZE MODEL LAYERS
        if self.freeze_method != 'none':
            self._freeze_all()
            if self.freeze_method == 'layer':
                self._unfreeze_by_layer(self)

        if self.feature_extractor is not None:
            self.feature_extractor_kwargs = {}
            if self.is_whisper_model:
                self.features_key = 'input_features'
                self.feature_extractor_kwargs['return_attention_mask'] = True
            else:
                self.features_key = 'input_values'

    def _unfreeze_by_layer(self):
        """
        Unfreeze specific model layers
        Needs to be overwritten depending on how each model stores layers
        """
        raise NotImplementedError('Unfreezing by layer not implemented for hugging face models')
    
    def forward(self, sample:Dict):
        if self.feature_extractor is not None:
            preprocessed_wav = self.feature_extractor(list(sample['waveform']).cpu().numpy(),
                                                      return_tensors='pt', 
                                                      sampling_rate = self.target_sample_rate,
                                                      **self.feature_extractor_kwargs)
            
            if self.is_whisper_model:
                output = self.base_model.encoder(preprocessed_wav[self.features_key].to(self.base_model.device))
            else:
                output = self.base_model(preprocessed_wav[self.features_key].to(self.base_model.device))
        else:
            output = self.base_model(sample['waveform'].to(self.base_model.device))

        pooled = self.pooling(output)

        return self.clf(pooled)
