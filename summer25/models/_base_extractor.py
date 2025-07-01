"""
Base feature extractor class

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORT
##built-in
from abc import abstractmethod
from pathlib import Path
from typing import Union, Optional

class BaseExtractor():
    """
    Base extractor

    :param model_type: str, type of model being initialized
    :param pt_ckpt: pathlike, path to base pretrained model checkpoint (default=None)
    """
    def __init__(self, model_type:str, pt_ckpt:Optional[Union[Path,str]]):
        self.model_type = model_type
        self.pt_ckpt = pt_ckpt
    
    @abstractmethod
    def _initialize_extractor(self):
        pass
    
    @abstractmethod
    def __call__(self) -> dict:
        pass