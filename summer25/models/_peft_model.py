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
from peft import PeftModel, PeftConfig
import torch 

class CustomPeft(PeftModel):
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
