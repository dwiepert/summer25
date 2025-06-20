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
import shutil
from typing import Union, Dict, Optional, List

##third-party
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModel, WhisperModel

##local
from ._base_model import BaseModel
from ._classifier import Classifier
from summer25.constants import _MODELS

class HFModel(BaseModel):
    """
    Model class for hugging face models 

    :param out_dir: Pathlike, output directory to save to
    :param model_type: str, hugging face model type for naming purposes
    :param freeze_method: str, freeze method for base pretrained model (default=required-only)
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
                 out_dir:Union[Path, str], model_type:str, freeze_method:str = 'required-only', 
                 pool_method:str = 'mean',pt_ckpt:Optional[Union[Path,str]]=None, ft_ckpt:Optional[Union[Path,str]]=None, 
                 out_features:int=1, nlayers:int=2, activation:str='sigmoid',
                 seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 from_hub:bool=True, test_hub_fail:bool=False, test_local_fail:bool=False, **kwargs):

        self.out_dir = out_dir 
        if not isinstance(self.out_dir, Path): self.out_dir = Path(self.out_dir)
        
        super().__init__(model_type=model_type, out_dir=out_dir,
                         freeze_method=freeze_method, pool_method=pool_method, 
                         pt_ckpt=pt_ckpt,ft_ckpt=ft_ckpt,
                         in_features=_MODELS[model_type]['in_features'], out_features=out_features, nlayers=nlayers, activation=activation, 
                         device=device, seed=seed,
                         pool_dim=_MODELS[model_type]['pool_dim'],**kwargs)

        #HF ARGS
        #handle some hugging face model specific parameters
        self.required_freeze = _MODELS[self.model_type]['required_freeze']
        self.optional_freeze = _MODELS[self.model_type]['optional_freeze']
        self.unfreeze_prefixes = _MODELS[self.model_type]['unfreeze_prefixes']

        assert 'hf_hub' in _MODELS[self.model_type], f'{self.model_type} is incompatible with HFModel class.'
        self.hf_hub = _MODELS[self.model_type]['hf_hub']

        if "delete_download" in kwargs:
            self.delete_download = kwargs.pop("delete_download")
        else:
            self.delete_download = False

        self.from_hub = from_hub

        if not self.from_hub: 
            assert self.pt_ckpt is not None, 'Must give pt_ckpt if not loading from the hub'
            assert self.pt_ckpt.exists(), 'Given pt_ckpt does not exist.'
        
        #INITIALIZE MODEL COMPONENTS
        print(f'Loading model {self.model_type} from Hugging Face Hub...')
        self._load_model(test_hub_fail=test_hub_fail, test_local_fail=test_local_fail)      
        self.is_whisper_model = isinstance(self.base_model, WhisperModel)
        self._load_checkpoint(self.ft_ckpt)
        self._freeze()
        self.base_model = self.base_model.to(self.device)

        # INITIALIZE CLASSIFIER (doesn't need to be overwritten)
        self.clf = Classifier(**self.clf_args)
        self.clf = self.clf.to(self.device)

        if self.local_path and not self.delete_download:
            self.base_config['pt_ckpt'] = str(self.local_path)

        self.model_name = self.get_model_name()
        self.config = {'model_name':self.model_name,'model_type':self.model_type,'seed':self.seed}
        
        self.config.update(self.base_config)
        self.config.update(self.clf.get_config())
        self.save_config()
    
    def get_model_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been finetuned
        Update for new model classes
        :return model_name: str with model name
        """
        model_name = f'{self.model_type}_seed{self.seed}_f{self.freeze_method}_{self.pool_method}'
        if self.ft_ckpt is not None:
            model_name += '_ft'
        return model_name

    def _load_model(self, test_hub_fail:bool=False, test_local_fail:bool=False):
        """
        Load pretrained model from hugging face

        :param test_hub_fail: bool, for testing purposes to confirm that non-hugging face functionality works (default=False)
        :param test_local_fail: bool, for testing purposes to confirm that failing a local load raises errors (default=False)
        """
        if self.from_hub:
            try: 
                if test_hub_fail or test_local_fail: 
                    raise Exception()
                self.base_model = AutoModel.from_pretrained(self.hf_hub, output_hidden_states=True, trust_remote_code=True)
                self.local_path = None
                return
            except:
                try:
                    if test_local_fail:
                        raise Exception()
                    
                    print('Loading directly from hugging face hub failed. Downloading model locally...')
                    self.local_path = Path(f'./{self.hf_hub}').absolute()
                    self.local_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(repo_id=self.hf_hub, local_dir=str(self.local_path))
                    self.base_model = AutoModel.from_pretrained(str(self.local_path), output_hidden_states=True)
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

        try:
            self.base_model = AutoModel.from_pretrained(str(self.pt_ckpt), output_hidden_states=True)
            self.local_path = self.pt_ckpt
        except:
            raise ValueError('Pretrained checkpoint is incompatible with HuggingFace models. Confirm this is a path to a local hugging face checkpoint.')
        
        return
                   
    def _freeze(self):
        """
        Freeze model with specified method
        """
        #FREEZE MODEL LAYERS
        self._freeze_all()
        
        if self.freeze_method == 'required-only': #only case where optional freeze layers should be unfrozen
            layer_names = self._get_unfreezable_layers(self.unfreeze_prefixes+self.optional_freeze)
        else:
            layer_names = self._get_unfreezable_layers(self.unfreeze_prefixes)
        
        #return layer names that can be unfrozen #only 
        if self.freeze_method == 'required-only' or self.freeze_method == 'optional':
            self.unfreeze = layer_names
        elif self.freeze_method == 'half':   
            if self.is_whisper_model:
                ind = int((len(layer_names)-1)/2) #calculation needs to only consider encoder.layers and not encoder.layer_norm 
                ind += 1
            else:
                ind = int(len(layer_names)/2)
            self.unfreeze = layer_names[-ind:]
        elif self.freeze_method == 'exclude-last':
            ind = -1
            if self.is_whisper_model: #calculations needs to exclude final encoder.layer_norm (unfreeze actual final encoder.layers) 
                ind -= 1
            self.unfreeze = layer_names[ind:]
        elif self.freeze_method == 'layer':
            self.unfreeze = self.unfreeze_layers

        self._unfreeze_by_layer(self.unfreeze) 

    def _get_unfreezable_layers(self, unfreezable_prefixes:List[str],group_level:int=1) -> List[str]:
        """
        Get layer names that can be unfrozen

        :param unfreezable: List[str], list of layer prefixes that can be unfrozen
        :param group_level: which layer group to return from model (based on unfreezable_prefixes) (default=1)
        :return layer_names: List[str], list of layers that can be unfrozen (group level down, e.g. if 'encoder.layers' in unfreezable_prefixes and group_level = 1, 'encoder.layers.0' in layer_names)
        """
        layer_names = []
    
        for name, _ in self.base_model.named_parameters():
            #print(name)
            for u in unfreezable_prefixes:
                if u in name:
                    p = u.split(".")
                    n = name.split(".")
                    if len(p) == len(n) or len(p)+1 == len(n):
                        if u not in layer_names:
                            layer_names.append(u)
                    else:
                        l = ".".join(n[:len(p)+group_level])
                        if l not in layer_names:
                            layer_names.append(l)

        return layer_names
    
    def forward(self, sample:Dict):
        """
        Overwritten forward loop. 

        :param sample: batched sample from a DataLoader object that minimally contains a `waveform` key storing the tensor of the loaded audio
        :return: classifier output
        """
        print('Where to use processor??? here ok??? or needs to be outside?')
        preprocessed_wav = sample['waveform']
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
    
