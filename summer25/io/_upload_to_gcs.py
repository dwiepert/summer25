"""
Upload file or directory to GCS bucket

Author(s): Daniela Wiepert
Last modified: 07/2025
"""

#IMPORTS
##built-in
from typing import Union, List
from pathlib import Path
##local
from ._search_gcs import search_gcs 

def upload_to_gcs(gcs_prefix:str, path:Union[str,Path], bucket, overwrite:bool=False, directory:bool=False) -> List[str]:
    '''
    Upload a file to a google cloud storage bucket
    Inputs:
    :param gcs_prefix: path in the bucket to save file to (no gs://project-name in the path)
    :param path: local string path of the file to upload
    :param bucket: initialized GCS bucket object 
    :param overwrite: bool, whether to overwrite (default = False)
    :param directory: bool, true if given gcs_prefix is a directory (default = True)
    :return to_upload: list of files to upload
    '''
    assert bucket is not None, 'no bucket given for uploading'
    #check if directory 
    if directory:
        gcs_pattern = '*'
    else:
        gcs_pattern = str(Path(path).name)
    
    gcs_prefix = str(gcs_prefix)
    if gcs_prefix[-1] != '/': gcs_prefix = gcs_prefix + '/'

    existing = search_gcs(gcs_pattern, gcs_prefix, bucket=bucket)
    
    if not isinstance(path, Path): path = Path(path)
    if path.is_dir():
        to_upload = [r for r in path.rglob('*') if not r.is_dir()]
    else:
        to_upload = [path]
    
    if not overwrite:
        keep = []
        for r in to_upload:
            if not any([str(r.name) in e for e in existing ]):
                keep.append(r)
        to_upload = keep

    for u in to_upload:  
        blob = bucket.blob(str(gcs_prefix + str(u.name)))
        blob.upload_from_filename(str(u))

    return to_upload 
