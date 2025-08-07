"""
Test I/O functions

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import json
import os
import shutil

##third-party
from google.cloud import storage
import pytest
from pathlib import Path
import torch
import torchaudio 

##local
from summer25.io import load_waveform_from_local, download_to_local, search_gcs, upload_to_gcs

##### HELPER FUNCTIONS#####
def load_json():
    with open('./private_loading/gcs.json', 'r') as file:
        data = json.load(file)

    gcs_prefix = data['gcs_prefix']
    bucket_name = data['bucket_name']
    project_name = data['project_name']
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    return gcs_prefix, bucket

##### TESTS #####
def test_load_local():
    #not structured
    input_dir=Path('./tests/audio_examples/')
    uid = '1919-142785-0008'
    extension = 'flac'
    waveform, sr = load_waveform_from_local(input_dir=input_dir, uid=uid,extension=extension, lib=False, structured=False)
    assert isinstance(waveform, torch.Tensor), 'Waveform not loaded'
    assert sr == 16000, 'Incorrectly loaded sample rate'

    #librosa
    waveform, sr = load_waveform_from_local(input_dir=input_dir, uid=uid,extension=extension, lib=True, structured=False)
    assert isinstance(waveform, torch.Tensor), 'Waveform not loaded'
    assert sr == 16000, 'Incorrectly loaded sample rate'

    #not existing audio 
    with pytest.raises(FileNotFoundError):
        uid = '1919-142785-0009'
        waveform, sr = load_waveform_from_local(input_dir=input_dir, uid=uid,extension=extension, lib=False, structured=False)
    
    #structured
    uid = '1919-142785-0008'
    temp_structured = Path(f'./tests/structured/{uid}')
    if temp_structured.exists():
        shutil.rmtree(temp_structured)
    if not temp_structured.exists():
        temp_structured.mkdir(parents=True)
    shutil.copy(f'./tests/audio_examples/{uid}.flac', f'./tests/structured/{uid}/waveform.flac')

    waveform, sr = load_waveform_from_local(input_dir=Path('./tests/structured'), uid=uid,extension=extension, lib=False, structured=True)
    assert isinstance(waveform, torch.Tensor), 'Waveform not loaded'
    assert sr == 16000, 'Incorrectly loaded sample rate'
    shutil.rmtree('./tests/structured')

@pytest.mark.gcs
def test_download_gcs():
    gcs_prefix, bucket = load_json()
    gcs_prefix += 'test_audio/'
    out_dir = Path('./out_dir')
    out_dir.mkdir(exist_ok=True)

    #download diretory
    with pytest.raises(AssertionError):
        download_to_local(gcs_prefix=gcs_prefix, savepath=out_dir, bucket=None, directory=True)

    files = download_to_local(gcs_prefix=gcs_prefix, savepath=out_dir, bucket=bucket, directory=True) #TODO: change gcs_path to gcs_prefix where using download to local
    #check what's in directory
    assert all([f.exists() for f in files])
    assert all([torchaudio.load(f)[0].numel() != 0])
    shutil.rmtree(out_dir)

    out_dir.mkdir(exist_ok=True)
    #download single file
    gcs_prefix += 'test1.wav'
    download_to_local(gcs_prefix=gcs_prefix, savepath=out_dir, bucket=bucket, directory=False)
    assert all([f.exists() for f in files])
    assert all([torchaudio.load(f)[0].numel() != 0])
    #check downloaded properly
    shutil.rmtree(out_dir)

@pytest.mark.gcs
def test_upload_gcs():
    gcs_prefix, bucket = load_json()
    gcs_prefix += 'test_upload_audio'
    to_delete = ['1919-142785-0008.flac','1919-142785-0060.flac']
    test_upload = 'tests/audio_examples/'

    upload_list = [str(Path(test_upload) / t) for t in to_delete]
    out_dir = Path('./out_dir')
    out_dir.mkdir(exist_ok=True)

    #clean out test dir just in case
    existing = search_gcs('*', gcs_prefix, bucket)
    for e in existing:
        blob = bucket.blob(e)
        blob.delete()
    existing = search_gcs('*', gcs_prefix, bucket)
    assert existing == []

    #upload directory 
    with pytest.raises(AssertionError):
        upload_to_gcs(gcs_prefix = gcs_prefix, path=test_upload, bucket=None)
    
    to_upload = upload_to_gcs(gcs_prefix = gcs_prefix, path=test_upload, bucket=bucket, overwrite=False, directory=True)
    assert all([str(t) in upload_list for t in to_upload])
    existing = search_gcs('*', gcs_prefix, bucket)
    for t in to_delete:
        assert any([t in e for e in existing])
    blob = bucket.blob(existing[0])
    blob.download_to_filename(str(out_dir / Path(existing[0]).name))
    assert torchaudio.load(str(out_dir / Path(existing[0]).name))[0].numel() != 0 
    os.remove(str(out_dir / Path(existing[0]).name))

    #try reuploading with ovewrite = True
    to_upload = upload_to_gcs(gcs_prefix = gcs_prefix, path=test_upload, bucket=bucket, overwrite=True, directory=True)
    assert all([str(t) in upload_list for t in to_upload])
    existing = search_gcs('*', gcs_prefix, bucket)
    for t in to_delete:
        assert any([t in e for e in existing])
    blob = bucket.blob(existing[0])
    blob.download_to_filename(str(out_dir / Path(existing[0]).name))
    assert torchaudio.load(str(out_dir / Path(existing[0]).name))[0].numel() != 0 
    os.remove(str(out_dir / Path(existing[0]).name))

    #try reuploading with overwrite = False 
    to_upload = upload_to_gcs(gcs_prefix = gcs_prefix, path=test_upload, bucket=bucket, overwrite=False, directory=True)
    assert to_upload == []
    existing = search_gcs('*', gcs_prefix, bucket)
    for t in to_delete:
        assert any([t in e for e in existing])
    blob = bucket.blob(existing[0])
    blob.download_to_filename(str(out_dir / Path(existing[0]).name))
    assert torchaudio.load(str(out_dir / Path(existing[0]).name))[0].numel() != 0 
    os.remove(str(out_dir / Path(existing[0]).name))

    # delete 
    for e in existing:
        blob = bucket.blob(e)
        blob.delete()
    existing = search_gcs('*', gcs_prefix, bucket)
    assert existing == []

    #upload single file
    test_upload += '1919-142785-0008.flac'
    to_upload = upload_to_gcs(gcs_prefix = gcs_prefix, path=test_upload, bucket=bucket, overwrite=False, directory=False)
    assert all([str(t) in upload_list for t in to_upload])
    existing = search_gcs('*', gcs_prefix, bucket)
    assert len(existing) == 1
    blob = bucket.blob(existing[0])
    blob.download_to_filename(str(out_dir / Path(existing[0]).name))
    assert torchaudio.load(str(out_dir / Path(existing[0]).name))[0].numel() != 0 
    os.remove(str(out_dir / Path(existing[0]).name))

    #try reuploading with ovewrite = True
    to_upload = upload_to_gcs(gcs_prefix = gcs_prefix, path=test_upload, bucket=bucket, overwrite=True, directory=False)
    assert all([str(t) in upload_list for t in to_upload])
    existing = search_gcs('*', gcs_prefix, bucket)
    assert len(existing) == 1
    blob = bucket.blob(existing[0])
    blob.download_to_filename(str(out_dir / Path(existing[0]).name))
    assert torchaudio.load(str(out_dir / Path(existing[0]).name))[0].numel() != 0 
    os.remove(str(out_dir / Path(existing[0]).name))

    #try reuploading with overwrite = False 
    to_upload = upload_to_gcs(gcs_prefix = gcs_prefix, path=test_upload, bucket=bucket, overwrite=False, directory=True)
    assert to_upload == []
    existing = search_gcs('*', gcs_prefix, bucket)
    assert len(existing) == 1
    blob = bucket.blob(existing[0])
    blob.download_to_filename(str(out_dir / Path(existing[0]).name))
    assert torchaudio.load(str(out_dir / Path(existing[0]).name))[0].numel() != 0 
    os.remove(str(out_dir / Path(existing[0]).name))

    # delete 
    for e in existing:
        blob = bucket.blob(e)
        blob.delete()
    existing = search_gcs('*', gcs_prefix, bucket)
    assert existing == []
    
    shutil.rmtree(out_dir)