from peft import PeftModel, PeftConfig
import torch 

class SpeechPeft(PeftModel):
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)

    def forward(
            self,
            inputs,
            **kwargs,
    ):
        peft_config = self.active_peft_config
        with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    inputs,
                    **kwargs,
                )
        