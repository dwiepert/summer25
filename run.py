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

##third-party
from torch.utils.data import DataLoader

##local
from summer25.models import HFModel, HFExtractor
from summer25.dataset import seeded_split, WavDataset, collate_features, collate_wrapper
from summer25.constants import _MODELS,_FREEZE, _FEATURES
from summer25.loops import finetune, evaluate
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
    if args.target_features:
        clf_args['out_features'] = len(args.target_features)
    else:
        clf_args['out_features'] = len(_FEATURES)
    if args.nlayers:
        clf_args['nlayers'] = args.nlayers
    if args.activation:
        clf_args['activation'] = args.activation
    if args.clf_ckpt:
        clf_args['ckpt'] = args.clf_ckpt
    return clf_args

def zip_extractor(args:argparse.Namespace) -> dict:
    """
    Zip arguments for extractor into a dictionary
    :param args: argparse Namespace object
    :return extractor_args: dict of model arguments
    """
    if args.hf_model:
        extractor_args = {'model_type':args.model_type}
    else:
        raise NotImplementedError('Only compatible with huggingface models currently.')
    
    if args.delete_download:
        extractor_args['delete_download'] = args.delete_download
    if args.pt_ckpt:
        extractor_args['pt_ckpt'] = args.pt_ckpt
    
    return extractor_args


def zip_model(args:argparse.Namespace) -> dict:
    """
    Zip arguments for base model into a dictionary
    :param args: argparse Namespace object
    :return model_args: dict of model arguments
    """
    if args.hf_model:
        model_args = {'model_type':args.model_type,'out_dir':args.output_dir,
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

    assert 'subject_key' in args
    assert 'date_key' in args
    assert 'task_key' in args
    assert 'audio_key' in args
    split_args['subject_key'] = args.subject_key
    split_args['date_key'] = args.date_key
    split_args['task_key'] = args.task_key
    split_args['audio_key'] = args.audio_key

    return split_args

def zip_dataset(args:argparse.Namespace) -> dict:
    """
    Zip arguments for dataset
    :param args: argparse Namespace object
    :return dataset_args: dict of dataset arguments
    """
    dataset_args = {'prefix': args.audio_dir, 'model_type': args.model_type, 'uid_col':args.audio_key, 'target_labels': args.target_features,
                    'bucket': None, 'extension': args.audio_ext, 'structured': args.structured}
    
    if args.transforms:
        transforms = args.transforms
    else:
        transforms = {}

    if args.use_librosa and 'use_librosa' not in transforms:
        transforms['use_librosa'] = args.use_librosa

    if 'truncate' not in transforms:
        if args.clip_length:
            truncate = {'length': args.clip_length}
            transforms['truncate'] = truncate
    
    if 'trim_level' not in transforms:
        if args.trim_level:
            transforms['trim_level'] = args.trim_level

    dataset_args['config'] = transforms
    return dataset_args

def zip_finetune(args):
    """
    Zip arguments for finetuning
    :param args: argparse Namespace object
    :return finetune_args: dict of finetune arguments
    """
    finetune_args = {'optim_type': args.optim_type, 'learning_rate':args.learning_rate, 'loss_type':args.loss_type,
                     'scheduler_type': args.scheduler_type, 'epochs': args.epochs, 'early_stop':args.early_stop,
                     'patience': args.patience, 'logging': args.logging}

    if args.end_lr:
        finetune_args['end_lr'] = args.end_lr

    return finetune_args

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
    io_args.add_argument('--audio_ext', type=str,default='wav', help='Audio extension.')
    io_args.add_argument('--structured', action='store_true', help='Specify whether the audio directory stores audio in structured manner (uid/waveform.wav) or not (uid.wav)')
    io_args.add_argument('--use_librosa', action='store_true', help='Specify whether to load audio with librosa')
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
    io_args.add_argument('--clip_length', type=float, help="Specify audio clip length in s.")
    io_args.add_argument('--trim_level', type=float, help="Specify silence trim level (dB for use_librosa, trigger level for torchaudio)")
    io_args.add_argument('--pad', action='store_true', help='Specify whether to pad audio by batch.')
    io_args.add_argument('--pad_method', default='mean', help='Specify whether to pad with 0s or mean of waveform')
    #BASE MODEL
    model_args = parser.add_argument_group('model', 'model related arguments')
    model_args.add_argument('--model_type', type=str,
                            choices=list(_MODELS.keys()), help='Specify model type')
    model_args.add_argument('--pt_ckpt', type=Path, help='Specify local pretrained model path. Only required for hugging face models if issues loading from hub.')
    model_args.add_argument('--delete_download', action='store_true', help='Specify local pretrained model path. Only required for hugging face models if issues loading from hub.')
    model_args.add_argument('--seed', type=int, help='Specify random seed for model initialization.')
    model_args.add_argument('--freeze_method', type=str, choices=_FREEZE, help='Specify what freeze method to use.')
    model_args.add_argument('--unfreeze_layers', nargs="+", help="If freeze_method is `layer`, use this to specify which layers to freeze")
    model_args.add_argument('--ft_ckpt', type=Path, help='Specify finetuned model checkpoint')
    model_args.add_argument('--pool_method', type=str, choices=['mean', 'max', 'attn'], help='Specify pooling method prior to classification head.')
    #CLASSIFIER
    clf_args = parser.add_argument_group('classifier', 'classifier related arguments')
    clf_args.add_argument('--nlayers', type=int, help='Specify classification head size.')
    clf_args.add_argument('--activation', type=str, help='Specify type of activation function to use in the classifier.')
    clf_args.add_argument('--clf_ckpt', type=Path, help="Specify classification checkpoint.")
    #TRAINING ARGS
    train_args = parser.add_argument_group('train', 'training/testing related arguments')
    train_args.add_argument('--batch_sz', default=1, type=int, help='Set batch size.')
    train_args.add_argument('--num_workers', default=0, type=int, help='Set num workers for DataLoader.')
    train_args.add_argument('--optim_type', type=str, default='adamw', help='Specify type of optimizer to use. (default = adamw)')
    train_args.add_argument('--learning_rate', type=float, default=1e-3, help='Specify learning rate (default=1e-3)')
    train_args.add_argument('--loss_type', type=str, default='bce', help='Specify type of optimizer to use. (default = bce)')
    train_args.add_argument('--scheduler_type', type=str, help='Specify type of scheduler to use.')
    train_args.add_argument('--end_lr', type=str, help='Specify end learning rate if using scheduelr')
    train_args.add_argument('--early_stop', action='store_true', help='Specify whether to use early stopping.')
    train_args.add_argument('--patience', type=int, default=5, help='Specify patience for early stopping. (default = 5)')
    train_args.add_argument('--logging', action='store_true', help='Specify whether to save logs.')
    train_args.add_argument('--epochs', type=int, default=1, help='Specify epochs for finetuning. (default = 1)')
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_l = check_load(args_dict)
    args_dict_m = check_model(args_dict_l)
    updated_args = save_path(argparse.Namespace(**args_dict_m))

    ## INITIALIZE EXTRACTOR AND MODEL
    ea = zip_extractor(updated_args)
    ma = zip_model(updated_args)
    ca = zip_clf(updated_args)
    ma.update(ca)
    sa = zip_splits(updated_args)
    da = zip_dataset(updated_args)
    fa = zip_finetune(updated_args) #don't forget extra scheduler args

    if args.hf_model:
        feature_extractor = HFExtractor(**ea)
        model = HFModel(**ma) #TODO: add sampling rate check
    else:
        raise NotImplementedError('Currently only compatible with hugging face models.')
    
    #DATA SPLIT
    train_df, val_df, test_df = seeded_split(**sa)

    #DATASET
    train_dataset = WavDataset(data=train_df,feature_extractor=feature_extractor, **da)
    test_dataset = WavDataset(data=test_df,feature_extractor=feature_extractor,**da)
    val_dataset = WavDataset(data=val_df, feature_extractor=feature_extractor,**da)
    
    #using custom collate
    if model.is_whisper_model:
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_sz,shuffle=True,collate_fn=collate_features, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_features, num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_features, num_workers=args.num_workers)
    else:
        collate_fn = collate_wrapper(pad=args.pad, pad_method=args.pad_method)
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_sz,shuffle=True,collate_fn=collate_fn, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_fn, num_workers=args.num_workers)
        test_aloader = DataLoader(dataset=test_dataset,batch_size=args.batch_sz,shuffle=False,collate_fn=collate_fn, num_workers=args.num_workers)
        
    if not args.eval_only:
        model = finetune(train_loader=train_loader, val_loader=val_load, model=model,
                         **fa)
    
    evaluate(test_loader=test_loader, model=model)

