"""
Custom AutoModel for loading pretrained/finetuned model checkpoints

Author(s): Daniela Wiepert
Last modified: 08/2025
"""
#IMPORTS
##built-in
from pathlib import Path
from typing import Union, Optional

##local
from summer25.constants import *
from summer25.io import search_gcs
from ._hf_model import HFModel


class CustomAutoModel:
    @classmethod
    def from_pretrained(cls, config:dict, ft_checkpoint:Optional[Union[str, Path]]=None, 
                        clf_checkpoint:Optional[Union[str,Path]]=None, pt_checkpoint:Optional[Union[str,Path]]=None,
                        delete_download:bool=False):
        """
        Load a model from a pretrained checkpoint
        :param config: dict, config of model arguments
        :param ft_checkpoint: pathlike, finetuned checkpoint path as a directory (default = None)
        :param clf_checkpoint: pathlike, classifier checkpoint path as a file (default = None)
        :param pt_checkpoint: pathlike, pretrained checkpoint path as a directorr (default = None)
        :param delete_download: bool, specify whether to delete any local downloads from hugging face (default = False)
        """

        def _split_ft_checkpoint(ft_checkpoint, clf_checkpoint, bucket, model_type):
            """
            Split a finetuned checkpoint directory into ft_checkpoint dir and clf_checkpoint path

            :param ft_checkpoint: pathlike, finetuned checkpoint path as a directory (default = None)
            :param clf_checkpoint: pathlike, classifier checkpoint path as a file (default = None)
            :param bucket: GCS bucket (default = None)
            :param model_type: str, model type

            :return ft_checkpoint: pathlike, split finetuned checkpoint path as a directory (default = None)
            :return clf_checkpoint: pathlike, split classifier checkpoint path as a file (default = None)
            """
            if ft_checkpoint:
                if not bucket:
                    if not isinstance(ft_checkpoint, Path): ft_checkpoint = ft_checkpoint(Path)
                    assert ft_checkpoint.is_dir(), 'Hugging face finetuned model checkpoints should be directories.'
                    #check first if there is a given clf_ckpt already
                    if clf_checkpoint is None:
                        poss_ckpts = [r for r in ft_checkpoint.glob('Classifier*')]
                        if poss_ckpts != []:
                            clf_checkpoint = poss_ckpts[0]
                    
                    base_ckpt = None
                    for entry in ft_checkpoint.iterdir():
                        if entry.is_dir() and model_type in str(entry):
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
                        for r in poss_ckpts:
                            if model_type in r:
                                ft_checkpoint = r
            return ft_checkpoint, clf_checkpoint
    


        if 'hf_hub' in _MODELS[config['model_type']]:
            model = HFModel(**config)
        else:
            mt = config['model_type']
            raise NotImplementedError(f'{mt} not implemented')
        
        assert pt_checkpoint or ft_checkpoint or model.from_hub, 'Must give a pretrained checkpoint, finetuned checkpoint, or load from hf hub.'

        # PEFT
        if model.peft:
            peft_delete_download = delete_download
            peft_from_hub = model.from_hub
            if model.finetune_method == 'soft-prompt':
                assert pt_checkpoint or peft_from_hub, 'Must give a pretrained checkpoint for model loading if using soft-prompt. Finetuned checkpoint only accounts for peft adapter.'
        else:
            peft_delete_download = False
            peft_from_hub = False

        #PRETRAINED
        pt_from_hub = model.from_hub
        pt_delete_download = delete_download if (not peft_delete_download or (ft_checkpoint and model.finetune_method == 'soft-prompt')) else False
        if pt_from_hub: #if from_hub, pt_checkpoint is always overridden with the hf_hub for the given model
            pt_checkpoint = model.hf_hub

        #EXTRACTOR
        ext_from_hub = model.from_hub
        ext_delete_download = delete_download if (ft_checkpoint and model.finetune_method != 'soft-prompt') else False 
        assert pt_checkpoint, 'Must have a pretrained option for loading extractor'

        #FINETUNED CASES
        ft_delete_download = delete_download if (not peft_delete_download) else False


        ### SPLIT FT CHECKPOINT INTO COMPONENTS IF REQUIRED
        ft_checkpoint, clf_checkpoint = _split_ft_checkpoint(ft_checkpoint, clf_checkpoint, model.bucket, model.model_type)

        ### load feature extractor using a pt_checkpoint
        model.load_feature_extractor(pt_checkpoint, ext_from_hub, ext_delete_download) 

        ### check if there is a ft_checkpoint AND it's not soft prompt
        if ft_checkpoint and model.finetune_method != 'soft-prompt':
            model.load_model_checkpoint(ft_checkpoint, ft_delete_download, False)
        else:
            model.load_model_checkpoint(pt_checkpoint, pt_delete_download, pt_from_hub)
        
        if clf_checkpoint:
            model.load_clf_checkpoint(clf_checkpoint, delete_download)

        if model.peft:
            if ft_checkpoint:
                model.configure_peft(ft_checkpoint, 'ft', peft_delete_download, False)
            else:
                model.configure_peft(pt_checkpoint, 'pt', peft_delete_download, peft_from_hub)
            
        model.save_config() #TODO

        model.to(model.device)
        return model 
    
    


    