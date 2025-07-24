"""
Peft model wrapper class for compatibility 

Author(s): Daniela Wiepert, peft source code
Last modified: 07/2025
"""
#IMPORTS
##built-in
import os
from typing import Union, Optional, Any
import warnings 
from accelerate.utils import named_module_tensors
from accelerate.hooks import remove_hook_from_submodules

##third-party
from peft import PeftModel, PeftConfig, PEFT_TYPE_TO_CONFIG_MAPPING, MODEL_TYPE_TO_PEFT_MODEL_MAPPING
import torch 


class SpeechPeft(PeftModel):
    """
    :param model: existing pytorch model
    :param peft_config: PeftConfig, config file for peft model
    :param adapter_name: str, adapter name (default = 'default)
    """
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)

    def forward(
            self,
            inputs:torch.Tensor,
            **kwargs,
    ):
        """
        Altered forward loop
        :param inputs: torch.Tensor, model input
        kwargs: additional forward loop arguments for different models
        """
        peft_config = self.active_peft_config
        with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                base_model = self.get_base_model()
                return base_model(
                    inputs,
                    **kwargs,
                )

## SOURCED FROM SOURCE PEFT
def peft_from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        key_mapping: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> SpeechPeft:
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
        ephemeral_gpu_offload (`bool`, *optional*):
            Whether to use ephemeral GPU offloading for partially loaded modules. Defaults to `False`. This is
            useful when parts of the model and/or components (such as adapters) are kept in CPU memory until they
            are needed. Rather than perform expensive operations on small data, the data is transferred to the GPU
            on-demand, the operation(s) performed, and the results moved back to CPU memory. This brings a slight
            momentary VRAM overhead but gives orders of magnitude speedup in certain cases.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device before loading the saved weights. Useful to speed up the
            process.
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
    if (key_mapping is None) and (not config.is_prompt_learning):
        key_mapping = getattr(model, "_checkpoint_conversion_mapping", {})

    # Runtime configuration, if supported
    if hasattr(config, "runtime_config"):
        config.runtime_config.ephemeral_gpu_offload = ephemeral_gpu_offload
    else:
        if ephemeral_gpu_offload:
            warnings.warn("Ephemeral GPU offloading is not supported for this model. Ignoring.")

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
    
    model = cls(
            model,
            config,
            adapter_name
        )
    
    load_result = model.load_adapter(
        model_id,
        adapter_name,
        is_trainable=is_trainable,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        key_mapping=key_mapping,
        **kwargs,
    )

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