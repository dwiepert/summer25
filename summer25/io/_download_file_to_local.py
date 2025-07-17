"""
Download file from GCS to local machine

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
from pathlib import Path
from typing import Union

##local
from ._search_gcs import search_gcs

def download_to_local(gcs_path: Union[str, Path], savepath: Union[str, Path], bucket) -> Path:
    """
    Download a single file to local path
    :param gcs_path: pathlike, current path in gcs
    :param savepath: pathlike, local path to save to
    :param bucket: GCS bucket with file
    :return savepath: Path, absolute savepath
    """
    file_blob = bucket.blob(str(gcs_path))
    files = search_gcs('*', gcs_path, bucket)
    if not isinstance(savepath,Path): savepath = Path(savepath)
    savepath = savepath.absolute()
    if len(files) == 1: 
        if not savepath.parents[0].exists():
            savepath.parents[0].mkdir(parents=True, exist_ok=True)
        
        file_blob.download_to_filename(str(savepath))
    else:
        for f in files:
            sub_path = savepath / Path(f).replace(str(gcs_path), "")
            if not sub_path.parents[0].exists():
                sub_path.parents[0].mkdir(parents=True, exist_ok=True)
            
            file_blob.download_to_filename(str(sub_path))

    return savepath

