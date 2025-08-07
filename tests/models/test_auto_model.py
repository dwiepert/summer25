"""
Test custom auto model loading

Author(s): Daniela Wiepert
Last modified: 08/2025
"""    
#IMPORTS 
##built-in
import json
import shutil
from pathlib import Path 

##third-party
from google.cloud import storage
from huggingface_hub import snapshot_download
import pytest

##local 
from summer25.models import CustomAutoModel
from summer25.io import search_gcs

### HELPER FUNCTIONS ###
def load_json():
    with open('./private_loading/gcs.json', 'r') as file:
        data = json.load(file)

    gcs_prefix = data['gcs_prefix']
    checkpoint_prefix = data['checkpoint_prefix']
    bucket_name = data['bucket_name']
    project_name = data['project_name']
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return gcs_prefix, checkpoint_prefix, bucket

def remove_gcs_directories(gcs_prefix,bucket, directory='test_split', pattern="*"):
    dir = gcs_prefix + f'{directory}'
    existing = search_gcs(pattern, dir, bucket)
    for e in existing:
        blob = bucket.blob(e)
        blob.delete()
    existing = search_gcs(pattern, dir, bucket)
    assert existing == []

### TESTS ###
@pytest.mark.hf
def download_checkpoint(checkpoint):
    local_path = Path(f'./test_checkpoints/{checkpoint}').absolute()
    local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=checkpoint, local_dir=str(local_path))
    return local_path
    
def test_auto_model():
    config = {'out_dir':Path('./test_checkpoints/outputs'), 'model_type':'test_model'}
    
    # give incompatible model type
    with pytest.raises(NotImplementedError):
        m = CustomAutoModel.from_pretrained(config)
    
    # no pt checkpoint, no ft checkpoint, from_hub = False
    config['model_type'] = 'wavlm-base'
    config['from_hub'] = False
    with pytest.raises(AssertionError):
        m = CustomAutoModel.from_pretrained(config)

    # load model from pt_checkpoint only
    # from_hub, pt_checkpoint is None
    config['from_hub'] = True
    m1 = CustomAutoModel.from_pretrained(config)
    assert m1.base_model is not None 

    # pt_checkpoint - regular 
    config['from_hub'] = False
    pt_checkpoint = download_checkpoint(m1.hf_hub)
    m2 = CustomAutoModel.from_pretrained(config, pt_checkpoint=pt_checkpoint)
    assert m2.base_model is not None
    clf_checkpoint = Path('./test_checkpoints') / 'clf_checkpoint.pt'
    m2._save_clf_checkpoint(clf_checkpoint)

    # pt_checkpoint - soft-prompt
    config['finetune_method'] = 'soft-prompt'
    m3 = CustomAutoModel.from_pretrained(config, pt_checkpoint=pt_checkpoint)
    assert m3.base_model is not None

    softprompt_ft_checkpoint = Path('./test_checkpoints') / 'test_softprompt'
    m3._save_model_checkpoint(softprompt_ft_checkpoint)

    # pt_checkpoint - lora
    config['finetune_method'] = 'lora'
    m4 = CustomAutoModel.from_pretrained(config, pt_checkpoint=pt_checkpoint)
    assert m4.base_model is not None 
    del config['finetune_method']

    lora_ft_checkpoint = Path('./test_checkpoints') / 'test_lora'
    m4._save_model_checkpoint(lora_ft_checkpoint)

    # load model from ft_checkpoint
    # pt_checkpoint is None, from_hub = False - FAIL
    ft_checkpoint = Path('./test_checkpoints') / 'test_checkpoint'
    m2._save_model_checkpoint(ft_checkpoint)
    config['from_hub'] = False
    with pytest.raises(AssertionError):
        m = CustomAutoModel.from_pretrained(config, ft_checkpoint=ft_checkpoint)

    # pt_checkpoint is None, from_hub = True
    del config['from_hub']
    m5 = CustomAutoModel.from_pretrained(config, ft_checkpoint=ft_checkpoint)
    assert m5.base_model is not None 

    # pt_checkpoint, from_hub = False
    config['from_hub'] = False 
    m6 = CustomAutoModel.from_pretrained(config, ft_checkpoint = ft_checkpoint, pt_checkpoint = pt_checkpoint)
    assert m6.base_model is not None 

    # lora, pt_checkpoint is None, from_hub = True
    del config['from_hub']
    config['finetune_method'] = 'lora'
    m7 = CustomAutoModel.from_pretrained(config, ft_checkpoint=lora_ft_checkpoint)
    assert m7.base_model is not None 

    # lora, pt_checkpoint, from_hub = False
    config['from_hub'] = False 
    m8 = CustomAutoModel.from_pretrained(config, ft_checkpoint=lora_ft_checkpoint, pt_checkpoint=pt_checkpoint)
    assert m8.base_model is not None 
        
    # soft-prompt, pt_checkpoint is None, from_hub = True
    del config['from_hub']
    config['finetune_method'] = 'soft-prompt'
    m9 = CustomAutoModel.from_pretrained(config, ft_checkpoint=softprompt_ft_checkpoint)
    assert m9.base_model is not None 

    # soft-prompt, pt_checkpoint, from_hub = False
    config['from_hub'] = False 
    m10 = CustomAutoModel.from_pretrained(config, ft_checkpoint=softprompt_ft_checkpoint, pt_checkpoint=pt_checkpoint)
    assert m10.base_model is not None 

    # give ft_checkpoint and clf_checkpoint separately
    del config['finetune_method']
    del config['from_hub']
    m11 = CustomAutoModel.from_pretrained(config, ft_checkpoint=ft_checkpoint, clf_checkpoint=clf_checkpoint)
    assert m11.base_model is not None 

    # give ft_checkpoint as directory with clf_checkpoint
    m2.save_model_components()
    m12 = CustomAutoModel.from_pretrained(config, ft_checkpoint=config['out_dir'])
    assert m12.base_model is not None 

    #give ft_checkpoint as parent dir and clf checkpoint separately
    m13 = CustomAutoModel.from_pretrained(config, ft_checkpoint=config['out_dir'], clf_checkpoint=clf_checkpoint)
    assert m13.base_model is not None 

    #give ft_checkpoint as parent dir and no clf_checkpoint exists 
    model_checkpoint2 = Path('./test_checkpoints/sub_test') / 'wavlm-base_model'
    m2._save_model_checkpoint(model_checkpoint2)
    m14 = CustomAutoModel.from_pretrained(config, ft_checkpoint=Path('./test_checkpoints/sub_test'))
    assert m14.base_model is not None 

    assert (config['out_dir'] / 'configs/model_config.json').exists()
    shutil.rmtree(Path('./test_checkpoints'))
    

#GCS VERSION - test delete download situations
#TODO
@pytest.mark.gcs
def test_auto_model_gcs():
    gcs_prefix, ckpt_prefix, bucket = load_json()
    config = {'out_dir':Path(f'{gcs_prefix}test_model'), 'model_type':'wavlm-base', 'bucket':bucket, 'from_hub':False}
    
    m = CustomAutoModel.from_pretrained(config, pt_checkpoint=ckpt_prefix, delete_download=True)
    assert m.base_model is not None
    assert not m.local_path.exists()

    m._save_model_checkpoint(f'{gcs_prefix}test_model/mdl_ckpt')
    m._save_clf_checkpoint(f'{gcs_prefix}test_model/clf_ckpt.pt')

    m1 = CustomAutoModel.from_pretrained(config, pt_checkpoint=ckpt_prefix, ft_checkpoint=f'{gcs_prefix}test_model/mdl_ckpt', delete_download=True)
    assert m1.base_model is not None
    assert not m1.local_path.exists()

    m2 = CustomAutoModel.from_pretrained(config, pt_checkpoint=ckpt_prefix, ft_checkpoint=f'{gcs_prefix}test_model/mdl_ckpt',  clf_checkpoint=f'{gcs_prefix}test_model/clf_ckpt.pt', delete_download=True)
    assert m2.base_model is not None
    assert not m2.local_path.exists()

    m3 = CustomAutoModel.from_pretrained(config, pt_checkpoint=ckpt_prefix, ft_checkpoint=f'{gcs_prefix}test_model',  delete_download=True)
    assert m3.base_model is not None
    assert not m3.local_path.exists()

    with pytest.raises(AssertionError):
        m = CustomAutoModel.from_pretrained(config, ft_checkpoint=f'{gcs_prefix}test_model',  delete_download=True)

    config['finetune_method'] = 'soft-prompt'
    m4 = CustomAutoModel.from_pretrained(config, pt_checkpoint=ckpt_prefix, ft_checkpoint=f'{gcs_prefix}test_model',  delete_download=True)
    assert m4.base_model is not None
    assert not m4.local_path.exists()

    with pytest.raises(AssertionError):
        m = CustomAutoModel.from_pretrained(config, ft_checkpoint=f'{gcs_prefix}test_model',  delete_download=True)

    config['finetune_method'] = 'lora'
    m5 = CustomAutoModel.from_pretrained(config, pt_checkpoint=ckpt_prefix, ft_checkpoint=f'{gcs_prefix}test_model',  delete_download=True)
    assert m5.base_model is not None
    assert not m5.local_path.exists()

    remove_gcs_directories(gcs_prefix, bucket, 'test_model')