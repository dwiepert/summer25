"""
Custom AutoModel for loading pretrained/finetuned model checkpoints

Author(s): Daniela Wiepert
Last modified: 07/2025
"""
#IMPORTS
##built-in
from accelerate.utils import named_module_tensors
from accelerate.hooks import remove_hook_from_submodules
from pathlib import Path
from typing import Union

##third-party
from peft import PeftModel, PeftConfig, PEFT_TYPE_TO_CONFIG_MAPPING, MODEL_TYPE_TO_PEFT_MODEL_MAPPING
import torch.nn as nn

##local
from summer25.constants import *
from ._hf_model import HFModel
from ._peft_model import CustomPeft


class CustomAutoModel:
    @classmethod
    def from_pretrained(cls, model_type:str, config:dict, ft_checkpoint:Optional[Union[str, Path]]=None, 
                        clf_checkpoint:Optional[Union[str,Path]]=None, pt_checkpoint:Optional[Union[str,Path]]=None, bucket=None,
                        delete_download:bool=False, test_hub_fail:bool=False, test_local_fail:bool=False) -> Union[HFModel, CustomPeft]:
        """
        Load a model from a pretrained checkpoint

        :param model_type: str, model type to load
        :param config: dict, config of model arguments
        :param ft_checkpoint: pathlike, finetuned checkpoint path as a directory (default = None)
        :param clf_checkpoint: pathlike, classifier checkpoint path as a file (default = None)
        :param pt_checkpoint: pathlike, pretrained checkpoint path as a directorr (default = None)
        :param bucket: GCS bucket (default = None)
        :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
        :param test_hub_fail: bool, TESTING ONLY (default = False)
        :param test_local_fail: bool, TESTING ONLY (default = False)
        """
        if 'hf_hub' in _MODELS[model_type]:
            model = HFModel(**config)
        else:
            raise NotImplementedError(f'{model_type} not implemented')
        
        ft_from_hub = model.from_hub 
        if ft_checkpoint:
            ft_from_hub = False     
        
        if model.from_hub: 
            pt_checkpoint = model.hf_hub
        else:
            assert pt_checkpoint or ft_checkpoint, 'Must give a pretrained checkpoint or finetuned checkpoint.'

        ### SPLIT FT CHECKPOINT INTO COMPONENTS IF REQUIRED
        ft_checkpoint, clf_checkpoint = self._split_ft_checkpoint(ft_checkpoint, clf_checkpoint, model.bucket)

        if pt_checkpoint:
            model.load_model_checkpoint(pt_checkpoint, delete_download, model.from_hub, test_hub_fail, test_local_fail)
        if ft_checkpoint:
            model.load_model_checkpoint(ft_checkpoint, delete_download, ft_from_hub, test_hub_fail, test_local_fail)
        if clf_checkpoint:
            model.load_clf_checkpoint(clf_checkpoint, delete_download)

        model.load_feature_extractor(pt_checkpoint, model.from_hub, delete_download)
        if model.peft:
            model.configure_peft(ft_checkpoint)
        
        model.save_config()

        model.to(model.device)
        return model 
    
    def _split_ft_checkpoint(self, ft_checkpoint, clf_checkpoint, bucket):
        """
        Split a finetuned checkpoint directory into ft_checkpoint dir and clf_checkpoint path

        :param ft_checkpoint: pathlike, finetuned checkpoint path as a directory (default = None)
        :param clf_checkpoint: pathlike, classifier checkpoint path as a file (default = None)
        :param bucket: GCS bucket (default = None)

        :return ft_checkpoint: pathlike, split finetuned checkpoint path as a directory (default = None)
        :return clf_checkpoint: pathlike, split classifier checkpoint path as a file (default = None)
        """
        if ft_checkpoint:
            if not bucket:
                if not isinstance(ft_ckpt, Path): ft_checkpoint = ft_checkpoint(Path)
                assert ft_checkpoint.is_dir(), 'Hugging face finetuned model checkpoints should be directories.'
                #check first if there is a given clf_ckpt already
                if clf_checkpoint is None:
                    poss_ckpts = [r for r in ft_ckpt.glob('Classifier*')]
                    if poss_ckpts != []:
                        clf_checkpoint = poss_ckpts[0]
                
                base_ckpt = None
                for entry in ft_checkpoint.iterdir():
                    if entry.is_dir():
                        base_ckpt = entry
                if base_ckpt is not None:
                    ft_checkpoint= base_ckpt
            else:
                existing = search_gcs(ft_checkpoint, ft_checkpoint, bucket)
                assert existing != [] and all([e != ft_checkpoint for e in existing]), 'Hugging face finetuned model checkpoints should be directories.'
                if clf_checkpoint is None:
                    poss_ckpts = [r for r in existing if 'Classifier' in r]
                    if poss_ckpts != []:
                        clf_checkpoint = poss_ckpts[0]

                poss_ckpts = list(set(["/".join(r.split("/")[:-1]) for r in existing if 'Classifier' not in r]))
                poss_ckpts = [r for r in poss_ckpts if r != ft_checkpoint]
                if poss_ckpts != []:
                    ft_checkpoint = poss_ckpts[0]
        return ft_checkpoint, clf_checkpoint
    
    @classmethod
    def peft_from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        **kwargs: Any,
    ) -> CustomPeft:
    r"""
    Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

    Note that the passed `model` may be modified inplace.

    Args:
        model ([`torch.nn.Module`]):
            The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
            [`~transformers.PreTrainedModel.from_pretrained`].
        model_id (`str` or `os.PathLike`):
            The name of the PEFT configuration to use. Can be either:
                - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                    Hub.
                - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                    method (`./my_peft_config_directory/`).
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter to be loaded. This is useful for loading multiple adapters.
        is_trainable (`bool`, *optional*, defaults to `False`):
            Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
            used for inference.
        config ([`~peft.PeftConfig`], *optional*):
            The configuration object to use instead of an automatically loaded configuration. This configuration
            object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
            loaded before calling `from_pretrained`.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Only relevant for specific adapter types.
        torch_device (`str`, *optional*, defaults to None):
            The device to load the adapter on. If `None`, the device will be inferred.
        key_mapping (dict, *optional*, defaults to None)
            Extra mapping of PEFT `state_dict` keys applied before loading the `state_dict`. When this mapping is
            applied, the PEFT-specific `"base_model.model"` prefix is removed beforehand and the adapter name (e.g.
            `"default"`) is not inserted yet. Only pass this argument if you know what you're doing.
        kwargs: (`optional`):
            Additional keyword arguments passed along to the specific PEFT configuration class.
    """
        #from .auto import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
        #from .tuners import XLoraConfig, XLoraModel

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        # See discussion in https://github.com/huggingface/transformers/pull/38627
        # Some transformers models can have a _checkpoint_conversion_mapping dict that is used to map state_dicts
        # stemming from updated model architectures so that they still correspond to the initial architecture. When
        # loading a PEFT state_dict created with the initial architecture on a model with the new architecture, we need
        # to map it too according to the same rules. Note that we skip prompt learning methods. This is because they
        # don't have the "base_model.model." prefix, which we need to remove before mapping. Instead just using
        # "base_model.". This could be fine, we could only remove "base_model.", However, the subsequent sub-module
        # could also be called "model", resulting in what looks like "base_model.model.". To avoid this confusion, we
        # skip prompt learning. Since it applies itself directly to the pre-trained model (unlike LoRA et al that target
        # sub-modules), skipping should be fine.

        key_mapping = getattr(model, "_checkpoint_conversion_mapping", {})

        if hasattr(model, "hf_device_map"):
            weight_map = dict(named_module_tensors(model, recurse=True))

            # recreate the offload_index for disk-offloaded modules: we need to know the location in storage of each weight
            # before the offload hook is removed from the model
            disk_modules = set()
            index = None
            for name, module in model.named_modules():
                if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "original_devices"):
                    if hasattr(module._hf_hook.weights_map, "dataset"):
                        index = module._hf_hook.weights_map.dataset.index
                    for key in module._hf_hook.original_devices.keys():
                        if module._hf_hook.original_devices[key] == torch.device("meta"):
                            disk_modules.add(str(name) + "." + str(key))

            if disk_modules and not kwargs.get("use_safetensors", True):
                raise ValueError("Disk offloading currently only supported for safetensors")

            if index:
                offload_index = {
                    p: {
                        "safetensors_file": index[p]["safetensors_file"],
                        "weight_name": p,
                        "dtype": str(weight_map[p].dtype).replace("torch.", ""),
                    }
                    for p in weight_map.keys()
                    if p in disk_modules
                }
                kwargs["offload_index"] = offload_index

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable
        
        model = CustomPeft(
                model,
                config,
                adapter_name
            )
        
        load_result = model.load_adapter(
            model_id,
            adapter_name,
            is_trainable=is_trainable,
            autocast_adapter_dtype=autocast_adapter_dtype,
            **kwargs,
        )
        
        ##REMOVED: key_mapping=key_mapping,

        # 1. Remove VB-LoRA vector bank, since it's a shared parameter set via the VBLoRAModel
        # 2. Remove the prompt encoder, as it does not need to be part of the checkpoint
        missing_keys = [
            k for k in load_result.missing_keys if "vblora_vector_bank" not in k and "prompt_encoder" not in k
        ]
        if missing_keys:
            # Let's warn here since (in contrast to load_adapter) we don't return the load result, so it could be quite
            # difficult for users to even notice that something might have gone wrong here. As we filter out non PEFT
            # keys from the missing keys, this gives no false positives.

            # careful: if the wording of the warning is changed, adjust the unit tests accordingly!
            warn_message = f"Found missing adapter keys while loading the checkpoint: {missing_keys}."

            warnings.warn(warn_message)

        return model


    