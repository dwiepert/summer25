"""
Download file from GCS to local machine

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
from pathlib import Path
from typing import Union, List

##local
from ._search_gcs import search_gcs

def download_to_local(gcs_prefix: Union[str, Path], savepath: Union[str, Path], bucket, directory:bool=False) -> List[Path]:
    """
    Download a single file to local path
    :param gcs_path: pathlike, current path in gcs
    :param savepath: pathlike, local path to save to
    :param bucket: GCS bucket with file
    :param directory: bool, true if given gcs_prefix is a directory (default = False)
    :return paths: List[Path], absolute savepaths for all files in directory
    """
    #file_blob = bucket.blob(str(gcs_path))
    #check if directory 
    assert bucket is not None, 'no bucket given for uploading'
    
    if directory:
        gcs_pattern = '*'
        gcs_prefix = str(gcs_prefix)
    else:
        gcs_pattern = str(Path(gcs_prefix).name)
        gcs_prefix = str(Path(gcs_prefix).parents[0]) + '/'

    files = search_gcs(gcs_pattern, gcs_prefix, bucket)
    
    if not isinstance(savepath,Path): savepath = Path(savepath).absolute()
    
    paths = []
    for f in files:
        sub_path = savepath / Path(f.replace(str(gcs_prefix), ""))
        if not sub_path.parents[0].exists():
            sub_path.parents[0].mkdir(parents=True, exist_ok=True)
        
        blob = bucket.blob(f)
        blob.download_to_filename(str(sub_path))
        paths.append(sub_path)

    return paths

