"""
Model class for hugging face models

Author(s): Daniela Wiepert
Last modified: 06/2025
"""

#IMPORT
##built-in
import os
from pathlib import Path
import shutil
from typing import Union, Optional, List
import warnings

##third-party
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, PeftModel, PromptTuningConfig, PromptTuningInit
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
    :param finetune_method: str, specify finetune method (default=None)
    :param freeze_method: str, freeze method for base pretrained model (default=required-only)
    :param unfreeze_layers: List[str], optionally give list of layers to unfreeze (default = None)
    :param pool_method: str, pooling method for base model output (default=mean)
    :param pt_ckpt: pathlike, path to pretrained model checkpoint (default=None)
    :param ft_ckpt: pathlike, path to finetuned base model checkpoint (default=None)
    :param clf_ckpt: pathlike, path to finetuned classifier checkpoint (default = None)
    :param out_features: int, number of output features from classifier (number of classes) (default = 1)
    :param nlayers: int, number of layers in classification head (default = 2)
    :param activation: str, activation function to use in classification head (default = 'relu')
    :param bottleneck: int, optional bottleneck parameter (default=None)
    :param layernorm: bool, true for adding layer norm (default=False)
    :param dropout: float, dropout level (default = 0.0)
    :param binary:bool, specify whether output is making binary decisions (default=True)
    :param lora_rank: int, optional value when using LoRA - set rank (default = 8)
    :param lora_alpha: int, optional value when using LoRA - set alpha (default = 16)
    :param lora_dropout: float, optional value when using LoRA - set dropout (default = 0.0)
    :param virtual_tokens: int, optional value when using soft prompting - set num tokens (default = 4)
    :param seed: int, specify random seed for ensuring reproducibility (default = 42)
    :param device: torch device (default = cuda)
    :param from_hub: bool, specify whether to load from hub or from existing pt_ckpt (default = True)
    :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
    :param test_hub_fail: bool, TESTING ONLY (default = False)
    :param test_local_fail: bool, TESTING ONLY (default = False)
    """
    def __init__(self, 
                 out_dir:Union[Path, str], model_type:str, finetune_method:str='none', freeze_method:str = 'required-only', unfreeze_layers:Optional[List[str]]=None,
                 pool_method:str = 'mean',pt_ckpt:Optional[Union[Path,str]]=None, ft_ckpt:Optional[Union[Path,str]]=None, clf_ckpt:Optional[Union[Path,str]]=None,
                 out_features:int=1, nlayers:int=2, activation:str='relu', bottleneck:int=None, layernorm:bool=False, dropout:float=0.0, binary:bool=True,
                 lora_rank:Optional[int]=8, lora_alpha:Optional[int]=16, lora_dropout:Optional[float]=0.0, virtual_tokens:Optional[int]=4,
                 seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 from_hub:bool=True, delete_download:bool=False, test_hub_fail:bool=False, test_local_fail:bool=False):

        self.out_dir = out_dir 
        if not isinstance(self.out_dir, Path): self.out_dir = Path(self.out_dir)
        
        #optionally split ft_ckpt into base_model ckpt and clf_ckpt
        if ft_ckpt:
            if not isinstance(ft_ckpt, Path): ft_ckpt = ft_ckpt(Path)
            assert ft_ckpt.is_dir(), 'Hugging face finetuned model checkpoints should be directories.'
            #check first if there is a given clf_ckpt already
            if clf_ckpt is None:
                poss_ckpts = [r for r in ft_ckpt.glob('Classifier*')]
                if poss_ckpts != []:
                    clf_ckpt = poss_ckpts[0]

            base_ckpt = None
            for entry in ft_ckpt.iterdir():
                if entry.is_dir():
                    base_ckpt = entry
            if base_ckpt is not None:
                ft_ckpt = base_ckpt
         
        super().__init__(model_type=model_type, out_dir=out_dir, finetune_method=finetune_method,
                         freeze_method=freeze_method, unfreeze_layers=unfreeze_layers, pool_method=pool_method, 
                         pt_ckpt=pt_ckpt,ft_ckpt=ft_ckpt, clf_ckpt=clf_ckpt,
                         in_features=_MODELS[model_type]['in_features'], out_features=out_features, nlayers=nlayers, activation=activation, 
                         bottleneck=bottleneck, layernorm=layernorm, dropout=dropout, binary=binary,
                         device=device, seed=seed, pool_dim=_MODELS[model_type]['pool_dim'])

        #HF ARGS
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.virtual_tokens = virtual_tokens
        if self.finetune_method == 'lora' or self.finetune_method == 'soft-prompt':
            self.freeze_method = 'all'
        #handle some hugging face model specific parameters
        self.required_freeze = _MODELS[self.model_type]['required_freeze']
        self.optional_freeze = _MODELS[self.model_type]['optional_freeze']
        self.unfreeze_prefixes = _MODELS[self.model_type]['unfreeze_prefixes']

        assert 'hf_hub' in _MODELS[self.model_type], f'{self.model_type} is incompatible with HFModel class.'
        self.hf_hub = _MODELS[self.model_type]['hf_hub']

        self.delete_download = delete_download

        self.from_hub = from_hub
        ckpt = None
        #CHECK IF FINETUNED CKPT
        if self.ft_ckpt:
            assert self.ft_ckpt.exists(), 'Given ft_ckpt does not exist.'
            assert self.ft_ckpt.is_dir(), 'Expects a directory for hugging face model checkpoints'
            #CHECK IF LORA FT CHECKPOINT
            if self.finetune_method == 'lora' or self.finetune_method == 'soft-prompt':
                #IF NOT FROM HUB, CHECK THAT PT CKPT EXISTS
                if not self.from_hub:
                    assert self.pt_ckpt is not None, 'Must give a pt checkpoint'
                    assert self.pt_ckpt.exists(), 'Given pt_ckpt does not exist.'
                    assert self.pt_ckpt.is_dir(), 'Expects a directory for hugging face model checkpoints'
                    ckpt = self.pt_ckpt
                #IF FROM HUB, DO NOTHING IT'S FINE
            else:
                #IF NOT LORA 
                ckpt = self.ft_ckpt 
                self.from_hub = False #must give false to load directly from FT checkpoint
        elif not self.from_hub or self.pt_ckpt:
            # IF NOT LOADING FROM HUB AND NOT FT MODEL PATH
            assert self.pt_ckpt is not None, 'Must give a pt checkpoint'
            assert self.pt_ckpt.exists(), 'Given pt_ckpt does not exist.'
            assert self.pt_ckpt.is_dir(), 'Expects a directory for hugging face model checkpoints'
            ckpt = self.pt_ckpt         

        #INITIALIZE MODEL COMPONENTS
        print(f'Loading model {self.model_type} from Hugging Face Hub...')

        self._load_model(ckpt=ckpt, test_hub_fail=test_hub_fail, test_local_fail=test_local_fail) 
        #print('Model parameters pre-freezing:')
        #trainable, allp = self._trainable_parameters(print_output=True)
        if not isinstance(self.base_model, _MODELS[self.model_type]['model_instance']):
            raise ValueError('Loaded model is not the expected model type. Please check that your checkpoint points to the correct model type.')
        self.is_whisper_model = isinstance(self.base_model, WhisperModel)
        self._freeze()
        if self.finetune_method == 'lora':
            self._configure_lora()
        elif self.finetune_method == 'soft-prompt':
            self._configure_softprompt()
        self.base_model = self.base_model.to(self.device)

        # INITIALIZE CLASSIFIER (doesn't need to be overwritten)
        self.clf = Classifier(**self.clf_args)
        self.clf = self.clf.to(self.device)

        if self.local_path and not self.delete_download:
            self.base_config['pt_ckpt'] = str(self.local_path)

        self.model_name = self.get_model_name()
        self.config = {'model_name':self.model_name,'model_type':self.model_type,'seed':self.seed, "finetune_method":self.finetune_method}
        
        self.config.update(self.base_config)
        self.config.update(self.clf.get_config())
        self.save_config()
    
    def get_model_name(self) -> str:
        """
        Get name for model type, including how it was freezed and whether it has been finetuned
        Update for new model classes
        :return model_name: str with model name
        """
        model_name = f'{self.model_type}_seed{self.seed}_{self.freeze_method}_{self.pool_method}'
        if self.ft_ckpt is not None:
            model_name += '_ft'
        if self.finetune_method != 'none':
            model_name += f'_{self.finetune_method}'
        return model_name

    def _load_model(self, ckpt, test_hub_fail:bool=False, test_local_fail:bool=False):
        """
        Load pretrained model from hugging face

        :param ckpt:
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
                    assert ckpt is not None, 'Downloading from hub failed, but backup pt_ckpt not available.'
                    print('Downloading from hub failed. Trying pt_ckpt.')

        try:
            self.base_model = AutoModel.from_pretrained(str(ckpt), output_hidden_states=True)
            self.local_path = ckpt
        except:
            raise ValueError('Checkpoint is incompatible with HuggingFace models. Confirm this is a path to a local hugging face checkpoint.')
        
        return
                   
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
    
    def _load_peft(self, ckpt:Union[str,Path]):
        """
        Load a peft model from a finetuned checkpoint
        :param ckpt:pathlike, path to finetuned checkpoint directory
        """
        if self.finetune_method == 'soft-prompt':
            is_trainable = False
            warnings.warn('Finetuned soft prompting models can not be loaded and further trained. Note that only classifier head will train if fitting a model.')
        else:
            is_trainable = True

        if ckpt:
            self.base_model = PeftModel.from_pretrained(self.base_model, 
                                                        ckpt,
                                                        is_trainable=is_trainable)

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
    
    def _configure_softprompt(self):
        """
        Configure soft prompting finetuning
        Creates soft prompting config and wraps the model using peft
        """
        print('Configuring soft prompting ...')
        if self.ft_ckpt:
            self._load_peft(self.ft_ckpt)
        else:
            if self.from_hub:
                ckpt = self.hf_hub
            else:
                ckpt = self.pt_ckpt
            peft_config = PromptTuningConfig(
                task_type="FEATURE_EXTRACTION", #This type indicates the model will generate text.
                prompt_tuning_init=PromptTuningInit.RANDOM,  #The added virtual tokens are initializad with random numbers
                num_virtual_tokens=self.virtual_tokens, #Number of virtual tokens to be added and trained.
                tokenizer_name_or_path=ckpt #The pre-trained model.
            )

            self.base_model = get_peft_model(self.base_model, peft_config)
        print('Using soft prompting: ')
        self.base_model.print_trainable_parameters()

    def _configure_lora(self):
        """
        Configure low rank adaptation finetuning
        Creates lora config and wraps the model using peft
        """
        print('Configuring LoRA ...')
        ## load previous
        if self.ft_ckpt:
            self._load_peft(self.ft_ckpt)
        else:
            #init_lora_weights="guassian"?
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

            self.base_model = get_peft_model(self.base_model, peft_config)
        print('Using LoRA: ')
        self.base_model.print_trainable_parameters()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Overwritten forward loop. 

        :param sample: batched sample feature input
        :return: classifier output
        """
        if self.is_whisper_model:
            output = self.base_model.encoder(x.to(self.device))
        else:
            output = self.base_model(x.to(self.device))
        output = output['last_hidden_state']
        
        pooled = self.pooling(output)
 
        return self.clf(pooled)
    
    def save_base_model(self, name:str, save_dir:Union[str,Path]):
        """
        Save base model with hugging face specific method
        :param name: str, name to save to
        :param save_dir: pathlike, directory to save to
        """
        if not isinstance(save_dir, Path): save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        save_dir = save_dir / name
        if save_dir.exists(): print('Overwriting existing base model file!')
        self.base_model.save_pretrained(save_dir)

    def save_model_components(self, name_prefix:str=None):
        """
        Save base model and classifier separately
        :param name_prefix: str, name prefix for model and classifier (default = None)
        """
        name_model = self.model_name
        name_clf = self.clf.config['clf_name']
        if name_prefix:
            name_model = name_prefix + name_model
            name_clf = name_prefix + name_clf
       
        out_path = self.out_dir / 'weights'
        out_path.mkdir(exist_ok=True)
        self.save_base_model(name_model, out_path)
        self.clf.save_classifier(name_clf, out_path)
    