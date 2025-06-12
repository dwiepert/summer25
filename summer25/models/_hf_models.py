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
    :param pt_ckpt: pathlike, path to pretrained model checkpoint (default=None)
    :param ft_ckpt: pathlike, path to finetuned base model checkpoint (default=None)
    :param out_features: int, number of output features from classifier (number of classes) (default = 1)
    :param nlayers: int, number of layers in classification head (default = 2)
    :param activation: str, activation function to use in classification head (default = 'sigmoid')
    :param seed: int, specify random seed for ensuring reproducibility
    :param device: torch device
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt
    :param test_hub_fail: bool, TESTING ONLY
    :param test_local_fail: bool, TESTING ONLY
    :param kwargs: additional arguments for optional parameters (e.g., unfreeze_layers if freeze_method is layer, delete_download if you want to remove local versions of checkpoints after downloading, clf_ckpt if wanting to load checkpoint for classifier)
    """
    def __init__(self, 
                 out_dir:Union[Path, str], model_type:str, keep_extractor:bool=True,
                 freeze_method:str = 'all', pool_method:str = 'mean',pt_ckpt:Optional[Union[Path,str]]=None, ft_ckpt:Optional[Union[Path,str]]=None, 
                 out_features:int=1, nlayers:int=2, activation:str='sigmoid',
                 seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 from_hub:bool=True, test_hub_fail:bool=False, test_local_fail:bool=False, **kwargs):

        self.out_dir = out_dir 
        if not isinstance(self.out_dir, Path): self.out_dir = Path(self.out_dir)
        
        #TODO: confirm weights are initialized properly - NEED TO ALSO DO THIS FOR THE CLASSIFIER!!!!!!!!!!
        super().__init__(model_type=model_type, out_dir=out_dir,
                         freeze_method=freeze_method, pool_method=pool_method, pt_ckpt=pt_ckpt,ft_ckpt=ft_ckpt,
                         in_features=_MODELS[model_type]['in_features'], out_features=out_features, nlayers=nlayers, activation=activation, 
                         device=device, seed=seed, base_only=False, pool_dim=_MODELS[model_type]['pool_dim'],**kwargs)

        #HF ARGS
        try:
            self.hf_hub = _MODELS[self.model_type]['hf_hub']
        except:
            assert False,  f'{self.model_type} is incompatible with HFModel class.'
        if "delete_download" in kwargs:
            self.delete_download = kwargs.pop("delete_download")
        else:
            self.delete_download = False
        #TODO: try this and see if it works
        self.keep_extractor = keep_extractor
        self.use_featext = _MODELS[self.model_type]['use_featext']
        self.from_hub = from_hub

        if not self.from_hub: 
            assert self.pt_ckpt is not None, 'Must give pt_ckpt if not loading from the hub'
            assert self.pt_ckpt.exists(), 'Given pt_ckpt does not exist.'
        
        if self.pt_ckpt is not None: #hugging face models don't require pt ckpt but could be used as a backup
            if not isinstance(self.pt_ckpt,str): self.pt_ckpt = str(self.pt_ckpt)

        #INITIALIZE MODEL COMPONENTS
        self._initialize_base_model()
        self.base_model = self.base_model.to(self.device, test_hub_fail=test_hub_fail, test_local_fail=test_local_fail)

        # INITIALIZE CLASSIFIER (doesn't need to be overwritten)
        self.clf = Classifier(**self.clf_args)
        self.clf = self.clf.to(self.device)
        #TODO: weight randomization for classifier

        if self.local_path and not self.delete_download:
            self.base_config['pt_ckpt'] = self.local_path

        self.config = {'model_name':self.get_model_name(),'model_type':self.model_type,'seed':self.seed, 'keep_extractor': self.keep_extractor}
        
        self.config.update(self._base_config)
        self.config_update(self.clf.get_config())
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

    def _load_model(self, test_hub_fail:bool=False, test_local_fail:bool=False):
        """
        TODO
        """
        if self.from_hub:
            try: 
                if test_hub_fail or test_local_fail: 
                    raise Exception()
                if self.use_featext:
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hf_hub, trust_remote_code=True)
                self.base_model = AutoModel.from_pretrained(self.hf_hub, output_hidden_states=True, trust_remote_code=True)
                self.local_path=None
                return 
            except:
                try:
                    if test_local_fail:
                        raise Exception()
                    
                    print('Loading directly from hugging face hub failed. Downloading model locally...')
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
                    return

                except: 
                    assert self.pt_ckpt is not None, 'Downloading from hub failed, but backup pt_ckpt not available.'
                    print('Downloading from hub failed. Trying pt_ckpt.')

        if self.use_featext:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.pt_ckpt)
        self.base_model = AutoModel.from_pretrained(self.pt_ckpt, output_hidden_states=True)
        return
                   
    def _initialize_base_model(self, test_hub_fail:bool=False, test_local_fail:bool=False):
        """
        Initialize base hugging face models
        TODO
        """
        print(f'Loading model {self.model_type} from Hugging Face Hub...')

        self._load_model(test_hub_fail=test_hub_fail, test_local_fail=test_local_fail)      
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
    
    def save_model_components(self):
        """
        Save base model and classifier separately
        """
        self.save_base_model()
        self.clf.save_classifier(self.out_dir)
