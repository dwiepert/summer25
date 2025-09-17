"""
Model class for hugging face models

Author(s): Daniela Wiepert
Last modified: 08/2025
"""

#IMPORT
##built-in
import gc
import os
from pathlib import Path
import shutil
from typing import Union, Optional, List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##third-party
from huggingface_hub import snapshot_download
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, WhisperModel

##local
from ._base_model import BaseModel
from ._hf_extractor import HFExtractor
from ._peft_model import CustomPeftModel
from ._classifier import Classifier
from summer25.constants import _MODELS
from summer25.io import upload_to_gcs, download_to_local, search_gcs

class HFModel(BaseModel):
    """
    Model class for hugging face models 

    :param out_dir: Pathlike, output directory to save to
    :param model_type: str, hugging face model type for naming purposes
    :param finetune_method: str, specify finetune method (default=None)
    :param freeze_method: str, freeze method for base pretrained model (default=required-only)
    :param unfreeze_layers: List[str], optionally give list of layers to unfreeze (default = None)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param normalize: bool, specify whether to normalize input
    :param out_features: int, number of output features from classifier (number of classes) (default = 1)
    :param nlayers: int, number of layers in classification head (default = 2)
    :param activation: str, activation function to use in classification head (default = 'relu')
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param binary:bool, specify whether output is making binary decisions (default=True)
    :param clf_type:str, specify layer type ['linear','transformer'] (default='linear')
    :param num_heads:int, number of encoder heads in using transformer build (default = 4)
    :param separate:bool, true if each feature gets a separate classifier head
    :param lora_rank: int, optional value when using LoRA - set rank (default = 8)
    :param lora_alpha: int, optional value when using LoRA - set alpha (default = 16)
    :param lora_dropout: float, optional value when using LoRA - set dropout (default = 0.0)
    :param virtual_tokens: int, optional value when using soft prompting - set num tokens (default = 4)
    :param seed: int, specify random seed for ensuring reproducibility (default = 42)
    :param device: torch device (default = cuda)
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
    :param print_memory: bool, true if printing memory information
    :param bucket: gcs bucket (default = None)
    """
    def __init__(self, 
                 out_dir:Union[Path, str], model_type:str, finetune_method:str='none', freeze_method:str = 'required-only', unfreeze_layers:Optional[List[str]]=None, pool_method:str = 'mean', normalize:bool=False,
                 out_features:int=1, nlayers:int=2, activation:str='relu', bottleneck:int=None, layernorm:bool=False, dropout:float=0.0, binary:bool=True, clf_type:str='linear', num_heads:int=4, separate:bool=True,
                 lora_rank:Optional[int]=8, lora_alpha:Optional[int]=16, lora_dropout:Optional[float]=0.0, virtual_tokens:Optional[int]=4,
                 seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 from_hub:bool=True, print_memory:bool=False, bucket=None):

        super().__init__(model_type=model_type, out_dir=out_dir, finetune_method=finetune_method,
                         freeze_method=freeze_method, unfreeze_layers=unfreeze_layers, pool_method=pool_method,
                         in_features=_MODELS[model_type]['in_features'], out_features=out_features, nlayers=nlayers, activation=activation, 
                         bottleneck=bottleneck, layernorm=layernorm, dropout=dropout, binary=binary, clf_type=clf_type, num_heads=num_heads, separate=separate,
                         device=device, seed=seed, pool_dim=_MODELS[model_type]['pool_dim'], bucket=bucket)

        #HF ARGS
        self.print_memory = print_memory
        self.normalize = normalize
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.virtual_tokens = virtual_tokens
        if self.finetune_method == 'lora' or self.finetune_method == 'soft-prompt':
            self.freeze_method = 'all'
            self.peft = True
        else:
            self.peft = False

        #handle some hugging face model specific parameters
        self.required_freeze = _MODELS[self.model_type]['required_freeze']
        self.optional_freeze = _MODELS[self.model_type]['optional_freeze']
        self.unfreeze_prefixes = _MODELS[self.model_type]['unfreeze_prefixes']

        assert 'hf_hub' in _MODELS[self.model_type], f'{self.model_type} is incompatible with HFModel class.'
        self.hf_hub = _MODELS[self.model_type]['hf_hub']
        
        if self.bucket:
            self.from_hub = False
        else:
            self.from_hub = from_hub      
        
        ## INITIALIZE CLASSIFIER ARCHITECTURE
        self.classifier_head = Classifier(**self.clf_args)

        ## INITIALIZE MODEL
        self.base_model = None

        ## INITIALIZE FEATURE EXTRACTOR
        #self.feature_extractor = None

        ## SET UP CONFIG
        self.model_name = self._get_model_name()
        self.config = {'model_name':self.model_name,'model_type':self.model_type}
        self.config.update(self.base_config)
        self.config.update(self.classifier_head.get_config())
        
    ### i/o functions (public and private) ###

    ##### CLASSIFIER #####
    def load_clf_checkpoint(self, checkpoint:Union[str,Path], delete_download:bool = False):
        """
        Load a checkpoint to a classifier head

        :param checkpoint: pathlike, path to checkpoint (.pth or .pt) 
        :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
        """

        if checkpoint is not None:
            if not isinstance(checkpoint, Path): checkpoint = Path(checkpoint)
            if not self.bucket:
                assert checkpoint.exists(), 'Given checkpoint does not exist.'
            else:
                existing = search_gcs(checkpoint, checkpoint, self.bucket)
                assert existing != [], 'Given checkpoint does not exist.'

            assert (checkpoint.suffix == '.pth') or (checkpoint.suffix == '.pt'), 'Must give path to .pth or .pt file for loading checkpoint.'
            try:
                if self.bucket:
                    save_path = Path('.')
                    files = download_to_local(checkpoint, save_path, self.bucket)
                    checkpoint = files[0]
               
                self.classifier_head.load_state_dict(torch.load(checkpoint, weights_only=True, map_location = self.device))
  
                if delete_download and self.bucket:
                    os.remove(checkpoint)

            except:
                if delete_download and self.bucket:
                    os.remove(checkpoint)

                raise ValueError('Classifier checkpoint could not be loaded. Weights may not be compatible with the initialized models.')

            self.config['clf_checkpoint'] = str(checkpoint)
    
    def _save_clf_checkpoint(self, path:Union[str,Path]):
        """
        Save classifier checkpoint

        :param path: pathlike, Path to save clf checkpoint at (FULL PATH)
        """
        if not isinstance(path, Path): path = Path(path)

        assert path.suffix == '.pth' or path.suffix == '.pt', 'Must give a full .pt or .pth filepath'

        if self.bucket:
            existing = search_gcs(path, path, self.bucket)
            if existing != []: print(f'Overwriting existing classifier head at {str(path)}')
            out_path = path
            path = Path('.') / out_path.name
        else:
            if path.exists(): print(f'Overwriting existing classifier head at {str(path)}')


        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.classifier_head.state_dict(), path)

        if self.bucket:
            upload_to_gcs(str(out_path.parent), path, self.bucket, overwrite=True, directory=False)
            os.remove(path)

    def load_model_checkpoint(self, checkpoint:Union[str,Path], delete_download:bool=False, from_hub:bool=True,
                              test_hub_fail:bool=False, test_local_fail:bool=False):
        """
        Load a model checkpoint

        :param checkpoint: pathlike, path to checkpoint (must be a directory)
        :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
        :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
        :param test_hub_fail: bool, TESTING ONLY (default = False)
        :param test_local_fail: bool, TESTING ONLY (default = False)
        """
        if from_hub:
            try: 
                print(f'Loading model {self.model_type} from Hugging Face Hub...')
                if test_hub_fail or test_local_fail: 
                    raise Exception()  
                self._configure_model(checkpoint)
                self.local_path = None
                return
            except:
                if self.base_model:
                    if not isinstance(self.base_model, _MODELS[self.model_type]['model_instance']):
                        raise ValueError('Loaded model is not the expected model type. Please check that your checkpoint points to the correct model type.')
                try:
                    if test_local_fail:
                        raise Exception()
                    
                    print('Loading directly from hugging face hub failed. Trying to download model locally...')
                    self.local_path = Path(f'./checkpoints/{checkpoint}').absolute()
                    self.local_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(repo_id=checkpoint, local_dir=str(self.local_path))
                    self._configure_model(checkpoint)
                    self._remove_download(self.local_path, delete_download, from_hub)
                    return
                except: 
                    print('Downloading from hub failed. Trying as local checkpoint.')

        if not self.bucket:
            if not isinstance(checkpoint, Path): checkpoint = Path(checkpoint)
            assert checkpoint.exists(), 'Given checkpoint does not exist.'
            assert checkpoint.is_dir(), 'Expects a directory for hugging face model checkpoints'
        else:
            existing = search_gcs(checkpoint, checkpoint, self.bucket)
            assert existing != []
            assert all([e != str(checkpoint) for e in existing]), 'Hugging face model checkpoints should be directories.'

        try:
            print(f'Loading model {self.model_type} from local checkpoint...')
            if self.bucket:
                local_path = Path('.')
                files = download_to_local(checkpoint, local_path, self.bucket, directory=True)
                checkpoint = files[0].parents[0].absolute()

            self.local_path = checkpoint
            self._configure_model(checkpoint)
            self._remove_download(self.local_path, delete_download, from_hub)

        except:
            self._remove_download(self.local_path, delete_download, from_hub)
            raise ValueError('Checkpoint is incompatible with HuggingFace models. Confirm this is a path to a local hugging face checkpoint.')

        self.config['model_checkpoint'] = str(checkpoint)
    
    # def configure_data_parallel(self, data_parallel:bool):
    #     """
    #     Make model compatible with multiple gpus

    #     :param data_parallel: bool, true if using data parallelization
    #     """
    #     assert self.base_model is not None, 'Model not loaded'
    #     if data_parallel:
    #         self.base_model = nn.DataParallel(self.base_model)
    #         self.classifier_head.configure_data_parallel(data_parallel)
        
    def _save_model_checkpoint(self, path:Union[str,Path]):
        """
        Save base model with hugging face specific method
        :param path: pathlike, directory to save model to
        """
        if not isinstance(path, Path): path = Path(path)
        assert path.suffix == '', 'Must give path to a directory'
        if not self.bucket:
            if path.exists(): print('Overwriting existing base model file!')
            path.parent.mkdir(exist_ok=True)
        else:
            out_path = path 
            path = Path('.') / path.name 
            existing = search_gcs(out_path, out_path, self.bucket)
            if existing != [] : print('Overwriting existing base model file!')
        
        self.base_model.save_pretrained(str(path))

        if self.bucket:
            upload_to_gcs(str(out_path), path, self.bucket, overwrite=True, directory=True)
            shutil.rmtree(path)

    def save_model_components(self, name_prefix:str=None, sub_dir:Path = None):
        """
        Save base model and classifier separately
        :param name_prefix: str, name prefix for model and classifier (default = None)
        :param sub_dir: Path, optional sub dir to save model components to
        """
        assert self.base_model is not None, 'Must have loaded base model'
        
        if sub_dir: 
            path = self.out_dir / sub_dir
        else:
            path = self.out_dir
        
        name_model = self.model_name
        name_clf = self.classifier_head.config['clf_name']
        if name_prefix:
            name_model = name_prefix + name_model
            name_clf = name_prefix + name_clf

        out_path = path / 'weights'

        if not self.bucket: out_path.mkdir(parents=True, exist_ok=True)
    
        self._save_model_checkpoint(path=out_path / name_model)
        self._save_clf_checkpoint(path= out_path / (name_clf + '.pt'))
    
    # def load_feature_extractor(self, checkpoint:Union[str,Path], from_hub:bool=True, delete_download:bool=False):
    #     """
    #     Load feature extractor from a checkpoint
    #     :param checkpoint: pathlike, path to checkpoint (must be a directory)
    #     :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
    #     :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
    #     """
    #     if from_hub: 
    #         bucket = None
    #     else:
    #         bucket = self.bucket

    #     self.feature_extractor = HFExtractor(model_type=self.model_type, pt_ckpt=checkpoint, from_hub=from_hub, normalize=self.normalize, bucket=bucket, delete_download=delete_download)
    #     #self.remove_download(self.local_path, delete_download, from_hub)

    def configure_peft(self, checkpoint:Union[str,Path], checkpoint_type:str='pt',delete_download:bool=False, from_hub:bool=True, load_gcs:bool=True):
        """
        Configure a Peft model 
        :param checkpoint: pathlike, path to checkpoint (must be a directory)
        :param checkpoint_type: str, specify whether finetuning (ft) or loading pretrained model (pt), (default = 'ft')
        :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
        :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
        :param load_gcs: bool, true if loading from gcs bucket
        """
        print(f'Configuring {self.finetune_method} ...')
        if self.bucket is None:
            load_gcs = False 

        if checkpoint_type == 'ft':
            self._load_peft_from_finetuned(checkpoint, delete_download, from_hub, load_gcs)
        else:
            if self.finetune_method == 'soft-prompt':
                peft_config = PromptTuningConfig(
                    task_type="FEATURE_EXTRACTION", #This type indicates the model will generate text.
                    prompt_tuning_init=PromptTuningInit.RANDOM,  #The added virtual tokens are initializad with random numbers
                    num_virtual_tokens=self.virtual_tokens, #Number of virtual tokens to be added and trained.
                    tokenizer_name_or_path=str(checkpoint) #The pre-trained model.
                )
            elif self.finetune_method == 'lora':
                assert 'lora_layers' in _MODELS[self.model_type], 'Model type incompatible with LoRA (no lora_layers specified).'
                if self.is_whisper_model:
                    exclude_modules = []
                    for id, (name, param) in enumerate(self.base_model.named_modules()):
                        if 'decoder' in name:
                            exclude_modules.append(name)
                else:
                    exclude_modules = None

                peft_config = LoraConfig(
                    r = self.lora_rank,
                    lora_alpha = self.lora_alpha,
                    target_modules=_MODELS[self.model_type]['lora_layers'],
                    exclude_modules = exclude_modules,
                    lora_dropout=self.lora_dropout,
                    bias="none", 
                    task_type="FEATURE_EXTRACTION"
                )
            else:
                raise NotImplementedError(f'{self.finetune_method} not implemented.')

            self.base_model = CustomPeftModel.peft_from_model(model=self.base_model, peft_config=peft_config)
            self._remove_download(checkpoint, delete_download, from_hub)

        print(f'Using {self.finetune_method}: ')
        self.base_model.print_trainable_parameters()
    
    def _load_peft_from_finetuned(self, ckpt:Union[str,Path], delete_download:bool=False, from_hub:bool=True, load_gcs:bool=True):
        """
        Load a peft model from a finetuned checkpoint
        :param ckpt:pathlike, path to finetuned checkpoint directory
        :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
        :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
        :param load_gcs: bool, true if loading from gcs bucket
        """
        if self.finetune_method == 'soft-prompt':
            is_trainable = False
            warnings.warn('Finetuned soft prompting models can not be loaded and further trained. Note that only classifier head will train if fitting a model.')
        else:
            is_trainable = True

        if load_gcs:
            local_path = Path('.')
            files = download_to_local(ckpt, local_path, self.bucket, directory=True)
            ckpt = files[0].parents[0].absolute()
            self.local_path = ckpt
            
            
        self.base_model = CustomPeftModel.peft_from_pretrained(model = self.base_model, 
                                                    model_id = ckpt,
                                                    is_trainable=is_trainable) #THERE IS GONNA BE AN ISSUE W THIS!! NEED TO MAKE MY OWN VERSION BOO

        self._remove_download(ckpt, delete_download, from_hub)
    ### private helper functions ###
    def _configure_model(self, checkpoint:Union[Path, str]):
        """
        Configure model using cutom AutoModel

        :param checkpoint: pathlike, path to checkpoint (must be a directory)
        """
        self.base_model = AutoModel.from_pretrained(checkpoint, output_hidden_states=True, trust_remote_code=True)       
        if not isinstance(self.base_model, _MODELS[self.model_type]['model_instance']):
            raise ValueError('Loaded model is not the expected model type. Please check that your checkpoint points to the correct model type.')
        self.is_whisper_model = isinstance(self.base_model, WhisperModel)
        self._freeze()

    def _remove_download(self, path:Union[str, Path], delete_download:bool=False, from_hub:bool=True):
        """
        Remove download

        :param path: pathlike, path to remove
        :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
        :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
        """
        if (self.bucket or from_hub) and delete_download:
            try:
                shutil.rmtree(str(path))

                bp = Path('.').absolute()
                curr_parent = Path(path).parent
                while bp.name != curr_parent.name: 
                    try:
                        os.rmdir(curr_parent)
                        temp = curr_parent.parent 
                        curr_parent = temp
                    except:
                        print('Parent directory not an empty directory')
            except:
                print('No download to delete.')
       
    def _get_model_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been finetuned
        Update for new model classes
        :return model_name: str with model name
        """
        model_name = f'{self.model_type}_seed{self.seed}_{self.freeze_method}_{self.pool_method}'
        if self.finetune_method != 'none':
            model_name += f'_{self.finetune_method}'
        return model_name
    
    def _freeze(self):
        """
        Freeze model with specified method
        """
        #FREEZE MODEL LAYERS
        self._freeze_all()
        self.unfreeze = []

        if self.freeze_method != 'all':
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

        #print('Model parameters post-freezing:')
        trainable, allp = self._trainable_parameters(print_output=False)
        
        if self.freeze_method == 'all':
            assert trainable == 0, 'Freezing did not work.'

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
    

    def _trainable_parameters(self, print_output:bool=False):
        """
        Calculate number of parameters, trainable or frozen

        :param print_output: bool, specify whether to print parameter counts
        :return trainable: int, number of trainable parameters
        :return allp: int, total number of parameters
        """
        allp = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        if print_output:
            print(f'trainable params: {trainable} || all params: {allp} || trainable%: {trainable/allp}')
        return trainable, allp
    
    def _downsample_attention_mask(self, attn_mask:torch.Tensor, target_len:int) -> torch.Tensor:
        """
        Downsample attention mask to target length

        :param attn_mask: torch.Tensor, attention mask
        :param target_len: int, target length to downsample 
        :return: downsampled attention mask
        """
        attn_mask = attn_mask.float().unsqueeze(1) # batch x 1 x time
        attn_mask = F.interpolate(attn_mask, size=target_len, mode="nearest")  # downsample
        return attn_mask.squeeze(1)
    
    def _check_memory(self, notice:str=None):
        """
        Check memory
        notice: str, notice to print
        """
        if self.print_memory:
            if notice:
                print(f'{notice}')
            c = torch.cuda.device_count()
            rm = [torch.cuda.memory_reserved(i) for i in range(c)]
            cm = [torch.cuda.memory_allocated(i) for i in range(c)]
            am = [rm[i]-cm[i] for i in range(c)]
            print(f'Reserved memory: {rm}')
            print(f'Current memory allocated:{cm}')
            print(f'Current memory available:{am}')
            
    ### main function(s) ###
    def forward(self, inputs: torch.Tensor, attention_mask:torch.Tensor) -> torch.Tensor:
        """
        Overwritten forward loop. 

        :param sample: batched sample feature input
        :return: classifier output
        """
        self._check_memory('Starting HF Model forward loop')
        
        #assert self.feature_extractor, 'Extractor checkpoints not loaded in.'
        assert self.base_model, 'Model checkpoints not loaded in.'

        #inputs, attention_mask = self.feature_extractor(waveform)
        #inputs = inputs.to(self.device)
        #attention_mask = attention_mask.bool().to(self.device)
        
        self._check_memory('Feature extractor finished. Inputs/attention mask sent to device. Geting base model outputs.')
    
        if self.is_whisper_model:
            output = self.base_model.encoder(inputs, attention_mask=attention_mask)
        else:
            output = self.base_model(inputs, attention_mask=attention_mask)

        del inputs
        self._check_memory('Model encoding retrieved. Starting pooling.')

        hs = output['last_hidden_state']
        del output 
        
        ds_attn_mask = self._downsample_attention_mask(attn_mask=attention_mask.to(torch.float16).detach(), target_len=hs.shape[1])
        expand_attn_mask = ds_attn_mask.unsqueeze(-1).repeat(1, 1, hs.shape[2])
        hs[~(expand_attn_mask==1.0)] = 0.0
        del attention_mask

        pooled = self._pool(hs, ds_attn_mask.to(self.device))
        del ds_attn_mask

        self._check_memory('Pooling finished. Sending to classifier.')

        clf_output = self.classifier_head(pooled)
        del pooled 
        
        self._check_memory('Classifier finished. Emptying cache.')
        
        gc.collect()
        #torch.cuda.empty_cache()
        
        self._check_memory()
        return clf_output

    