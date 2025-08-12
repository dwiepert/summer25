"""
Search gcs bucket for existing files

Author(s): Daniela Wiepert
Last modified: 07/2025
"""
#IMPORTS
##built-in
from pathlib import Path
from typing import Union

def search_gcs(pattern:Union[str,Path], directory:Union[str,Path], bucket, exact_match:bool=False) -> list:
    """
    Search gcs bucket based on prefix. 
    :param pattern: str, pattern to search for
    :param directory: str, directory to search
    :param bucket: gcs bucket
    :param exact_match: bool, True if looking for exact match
    :return: list of files
    """
    if not isinstance(pattern, str): pattern = str(pattern)
    if not isinstance(directory, str): directory = str(directory)
    files = []
    blobs = bucket.list_blobs(prefix=directory)
    for blob in blobs:
        name = blob.name
        if exact_match:
            s_name = Path(name).parts
            if any([s == pattern for s in s_name]) or pattern == name: #either a subpart matches or it's a full path match
                files.append(name)
        else:
            if (pattern in name) or (pattern == '*' and name != directory):
                files.append(name)
    
    return files