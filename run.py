"""
Run experiments

Author(s): Daniela Wiepert
Last modified: 06/2025
"""
#IMPORTS
##built-in
import argparse
import json
from pathlib import Path
from typing import List

##local
from summer25.models import HFModel
from summer25.dataset import seeded_split
from summer25.constants import _MODELS

_REQUIRED_MODEL_ARGS =['model_type']
_REQUIRED_LOAD = ['output_dir', 'audio_dir', 'split_dir']

# HELPER FUNCTIONS #
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# CHECK ARGUMENTS #
def check_load(args:dict) -> dict:
    """
    Load config file for data saving and audio loading, and ensure required arguments are there and pass assertions. 

    :param args: dictionary of arguments - vars(argparse object)
    :return args: updated args dictionary
    """ 
    if 'load_cfg' in args:
        load_cfg = args.pop('load_cfg')
        args.update(load_cfg)

    for r in _REQUIRED_LOAD:
        assert r in args, f'The required argument `{r}` was not given. Use `-h` or `--help` if information on the argument is needed.'

    assert args['audio_dir'], 'Audio directory not given.'
    assert args['split_dir'], 'Split dir not given.'
    assert args['output_dir'], 'Output directory not given.'

    if not isinstance(args['audio_dir'], Path): args['audio_dir'] = Path(args['audio_dir'])
    if not isinstance(args['split_dir'],Path): args['split_dir'] = Path(args['split_dir'])
    if not isinstance(args['output_dir'], Path): args['output_dir'] = Path(args['output_dir'])
    
    assert args['audio_dir'].exists(), 'Given audio directory does not exist.'
    args['split_dir'].mkdir(parents=True, exist_ok=True)
    args['output_dir'].mkdir(parents=True, exist_ok=True)
    
    return args

def check_model(args:dict ) -> dict:
    """
    Check if a model config file exists or has been given, load in if it does, and ensure required arguments are there and pass assertions. 

    If the model config file is not an existing model config (as saved in the model class), make sure that it follows the format of those config files. See README for more information. TODO!

    :param args: dictionary of arguments - vars(argparse object)
    :return args: updated args dictionary
    """
    #check for mismatch between existing and not existing model cfg
    ec = [e for e in args['output_dir'].rglob('*model_config.json')] 
    if ec != [] and args['use_existing_cfg']:
        existing_cfg = load_config(ec[0])
    else:
        existing_cfg = None
        #raise NotImplementedError('Updating args with existing config is not implemented.')

    #TODO: compare existing model_cfg and given arguments? or just overwrite with existing arguments? 

    if 'model_cfg' in args:
        model_cfg = args.pop('model_cfg')

        if 'clf_ckpt' in model_cfg:
            model_cfg['clf_ckpt'] = model_cfg.pop('clf_ckpt')
        args.update(model_cfg)

    if existing_cfg is not None:
        args.update(existing_cfg)
        
    for r in _REQUIRED_MODEL_ARGS:
        assert r in args, f'The required argument `{r}` was not given in a config file or in the command line. Use `-h` or `--help` if information on the argument is needed.'

    if args['clf_ckpt'] is not None:
        if not isinstance(args['clf_ckpt'], Path): args['clf_ckpt'] = Path(args['clf_ckpt'])
    if args['ft_ckpt'] is not None:
        if not isinstance(args['ft_ckpt'], Path): args['ft_ckpt'] = Path(args['ft_ckpt'])
    
    args['hf_model'] = False
    if 'hf_hub' in _MODELS[args['model_type']]: args['hf_model'] = True
        
    assert 'freeze_method' in args, 'Freeze method not given. Check command line arguments or model configuration file.'
    if args['freeze_method'] == 'layer': assert 'unfreeze_layers' in args, 'unfreeze_layers must be given if freeze method is `layer`.'
    return args

def save_path(args:argparse.Namespace) -> argparse.Namespace:
    """
    Generate save path to ensure models are not overwritten 
    """
    print('Output directory save path not fully implemented.')
    return args

# PREP ARGUMENTS #
def zip_clf(args:argparse.Namespace) -> dict:
    """
    Zip arguments for classifier into a dictionary
    :param args: argparse Namespace object
    :return clf_args: dict of classifier arguments
    """
    clf_args = {}
    if args.out_features:
        clf_args['out_features'] = args.out_features
    if args.nlayers:
        clf_args['nlayers'] = args.nlayers
    if args.activation:
        clf_args['activation'] = args.activation
    if args.clf_ckpt:
        clf_args['ckpt'] = args.clf_ckpt
    return clf_args

def zip_model(args:argparse.Namespace) -> dict:
    """
    Zip arguments for base model into a dictionary
    :param args: argparse Namespace object
    :return model_args: dict of model arguments
    """
    if args.hf_model:
        model_args = {'model_type':args.model_type,'out_dir':args.output_dir,'keep_extractor':args.keep_extractor, 
                    'freeze_method':args.freeze_method, 'pool_method':args.pool_method,
                    'seed':args.seed}
    else:
        raise NotImplementedError('Only compatible with huggingface models currently.')
    
    if args.delete_download:
        model_args['delete_download'] = args.delete_download
    if args.pt_ckpt:
        model_args['pt_ckpt'] = args.pt_ckpt
    if args.ft_ckpt:
        model_args['ft_ckpt'] = args.ft_ckpt
    if args.unfreeze_layers:
        model_args['unfreeze_layers'] = args.unfreeze_layers

    return model_args

def zip_splits(args:argparse.Namespace) -> dict:
    """
    Zip arguments for splits into a dictionary
    :param args: argparse Namespace object
    :return split_args: dict of split arguments
    """
    split_args = {'audio_dir': args.audio_dir, 'split_dir': args.split_dir,
                  'seed': args.seed, 'save': args.save_split, 'load_existing': args.load_existing_split, 'as_json':args.as_json}

    if args.proportions:
        split_args['proportions'] = args.proportions
    if args.target_tasks:
        split_args['target_tasks'] = args.target_tasks
    if args.target_features:
        split_args['target_features'] = args.target_features
    if args.stratify_threshold:
        split_args['stratify_threshold'] = args.stratify_threshold
    if args.subject_key:
        split_args['subject_key'] = args.subject_key
    if args.date_key:
        split_args['date_key'] = args.date_key
    if args.task_key:
        split_args['task_key'] = args.task_key
    if args.audio_key:
        split_args['audio_key'] = args.audio_key

    return split_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Parser")
    #OPTIONAL CONFIG FILES
    cfg_args = parser.add_argument_group('cfg', 'configuration file related arguments')
    cfg_args.add_argument('--load_cfg', type=load_config, help='Audio loading configuration json')
    cfg_args.add_argument('--model_cfg', type=load_config, help="Model configuration json")
    cfg_args.add_argument('--use_existing_cfg', action='store_true', help='Specify whether to use an existing config file if it exists in the given output_dir')
    #I/O
    io_args = parser.add_argument_group('io', 'file related arguments')
    io_args.add_argument('--audio_dir', type=Path, help='Directory with audio files & a csv with information on speakers/task.')
    io_args.add_argument('--split_dir', type=Path, help='Directory with csv or jsons of file splits.')
    io_args.add_argument('--load_exisiting_split', action='store_true', help='Default to loading exisiting split.')
    io_args.add_argument('--as_json', action='store_true', help='True if loading/saving splits as json files.')
    io_args.add_argument('--save_split', action='store_true', help='Save generated data splits.')
    io_args.add_argument('--output_dir', type=Path, help='Output directory for saving all files.')
    io_args.add_argument('--target_tasks', nargs="+", type=List[str], help="Specify target tasks for dataset.")
    io_args.add_argument('--target_features', nargs="+", type=List[str], help="Specify target features for dataset.")
    io_args.add_argument('--stratify_threshold', type=int, help="Specify minimum number of positive examples for a feature group.")
    io_args.add_argument('--subject_key', type=str, help='Specify column/key name for subjects in dataset metadata table')
    io_args.add_argument('--date_key', type=str, help='Specify column/key name for date in dataset metadata table')
    io_args.add_argument('--task_key', type=str, help='Specify column/key name for tasks in dataset metadata table')
    io_args.add_argument('--audio_key', type=str, help='Specify column/key name for audio file names in dataset metadata table')
    io_args.add_argument('--proportions', nargs="+", type=List[float], help='Specify split proportions.')
    #AUDIO
    audio_args = parser.add_argument_group('audio', 'Audio file related args')
    #BASE MODEL
    model_args = parser.add_argument_group('model', 'model related arguments')
    model_args.add_argument('--model_type', type=str,
                            choices=list(_MODELS.keys()), help='Specify model type')
    model_args.add_argument('--pt_ckpt', type=Path, help='Specify local pretrained model path. Only required for hugging face models if issues loading from hub.')
    model_args.add_argument('--delete_download', action='store_true', help='Specify local pretrained model path. Only required for hugging face models if issues loading from hub.')
    model_args.add_argument('--seed', type=int, help='Specify random seed for model initialization.')
    model_args.add_argument('--keep_extractor', action='store_true', help='Specify whether to keep weights of feature extractor.')
    model_args.add_argument('--freeze_method', type=str, choices=['all', 'layer', 'none'], help='Specify what freeze method to use.')
    model_args.add_argument('--unfreeze_layers', nargs="+", help="If freeze_method is `layer`, use this to specify which layers to freeze")
    model_args.add_argument('--ft_ckpt', type=Path, help='Specify finetuned model checkpoint')
    model_args.add_argument('--pool_method', type=str, choices=['mean', 'max', 'attn'], help='Specify pooling method prior to classification head.')
    #CLASSIFIER
    clf_args = parser.add_argument_group('classifier', 'classifier related arguments')
    clf_args.add_argument('--out_features', type=int, help='Specify number of classification categories (output features).')
    clf_args.add_argument('--nlayers', type=int, help='Specify classification head size.')
    clf_args.add_argument('--activation', type=str, help='Specify type of activation function to use in the classifier.')
    clf_args.add_argument('--clf_ckpt', type=Path, help="Specify classification checkpoint.")
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_l = check_load(args_dict)
    args_dict_m = check_model(args_dict_l)
    updated_args = save_path(argparse.Namespace(**args_dict_m))

    s = zip_splits(updated_args)
    train_df, val_df, test_df = seeded_split(**s)

    ma = zip_model(updated_args)
    ca = zip_clf(updated_args)
    ma.update(ca)

    if args.hf_model:
        model = HFModel(**ma) #TODO: add sampling rate check
    else:
        raise NotImplementedError('Currently only compatible with hugging face models.')
    print('pause')
