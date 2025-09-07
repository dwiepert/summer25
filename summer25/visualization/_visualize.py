import json 
import tempfile
from pathlib import Path
from typing import Union
from summer25.io import search_gcs, download_to_local 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

def get_data(directory:Union[str,Path], bucket=None):
    """
    """
    patterns = ['model_config.json', 'train_config.json', 'train_log.json', 'evaluation.json']

    data = {}
    if bucket:
        for p in patterns:
            files = search_gcs(p, directory, bucket)
            assert len(files) == 1
            with tempfile.TemporaryDirectory() as tmpdirname:
                paths = download_to_local(files[0], tmpdirname, bucket, True)
                with open(str(paths[0]), 'r') as j:
                    d = json.load(j)
                data[p] = d 
    else:
        for p in patterns:
            if not isinstance(directory, Path): directory = Path(directory)
            files = [f for f in directory.glob(f'*{p}')]
            assert len(files)==1
            with open(str(files[0]), 'r') as j:
                d = json.load(j)
            data[p] = d
    return data

def extract_data_from_parentdir(parent_directory:Union[str,Path], bucket=None):
    """
    """
    if not isinstance(parent_directory, Path): parent_directory = Path(parent_directory)
    if bucket:
        files = search_gcs(parent_directory, parent_directory, bucket)
        files = [Path(f) for f in files]
        sub_dirs = [f for f in files if str(f.parent) == str(parent_directory)]
        pass
    else:
        sub_dirs = [d for d in parent_directory.iterdir() if (str(d.parent) == str(parent_directory) and d.is_dir())]

    data = {}
    for s in sub_dirs:
        data[s] = get_data(s, bucket)
    
    return data

def plot_training_loss(data, save_dir):
    """
    """    
    colors = list(mcolors.TABLEAU_COLORS)
    r_colors = random.sample(colors, len(data))
    i = 0
    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr= train_md['tf_learning_rate']
        lr = train_md['learning_rate']


        plt.plot(output['train_log.json']['avg_train_loss'], label= f'Training loss - tf_lr{tflr}; lr{lr}', color=r_colors[i])
        plt.plot(output['train_log.json']['avg_val_loss'], label= f'Val loss - tf_lr{tflr}; lr{lr}', linestyle='--', color=r_colors[i])
        i += 1

    plt.title('Average loss during training')
    plt.xlabel('Avg. loss')
    plt.ylabel('Epochs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(str(Path(save_dir) / 'train_val_loss.png'), bbox_inches='tight', dpi=300)
    plt.close()

    i=0
    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr= train_md['tf_learning_rate']
        lr = train_md['learning_rate']


        plt.plot(output['train_log.json']['avg_train_loss'], label= f'Training loss - tf_lr{tflr}; lr{lr}', color=r_colors[i])
        i += 1

    plt.title('Average loss during training')
    plt.xlabel('Avg. loss')
    plt.ylabel('Epochs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(str(Path(save_dir) / 'training_loss.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_auc(data):
    """
    """
    pass
    
def plot_accuracy(data):
    """
    """
    pass
