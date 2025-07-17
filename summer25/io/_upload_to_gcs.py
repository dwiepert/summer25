#IMPORTS
##built-in
from typing import Union
from pathlib import Path
##local
from ._search_gcs import search_gcs 

def upload_to_gcs(gcs_prefix:str, path:Union[str,Path], bucket=None, overwrite:bool=False):
    '''
    Upload a file to a google cloud storage bucket
    Inputs:
    :param gcs_prefix: path in the bucket to save file to (no gs://project-name in the path)
    :param path: local string path of the file to upload
    :param bucket: initialized GCS bucket object
    :param overwrite: bool, whether to overwrite (default = False)
    '''
    assert bucket is not None, 'no bucket given for uploading'
    blob = bucket.blob(str(gcs_prefix))
    existing = search_gcs('*', directory= gcs_prefix, bucket=bucket)
    if path.is_dir():
        to_upload = [r for r in path.rglob('*') if not r.is_dir()]
    else:
        to_upload = [path]
    
    if not overwrite:
        to_upload = [r for r in to_upload if r not in existing]

    for u in to_upload:
        blob.upload_from_filename(str(u))

