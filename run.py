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

# HELPER FUNCTIONS #
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# CHECK ARGUMENTS #
def check_load(args:dict, required:List[str]=['output_dir']) -> dict:
    """
    Load config file for data saving and audio loading, and ensure required arguments are there and pass assertions. 

    :param args: dictionary of arguments - vars(argparse object)
    :param required: list of required arguments
    :return args: updated args dictionary
    """ 
    if 'load_cfg' in args:
        load_cfg = args.pop('load_cfg')
        args.update(load_cfg)
    else:
        for r in required:
            assert r in args, f'The required argument `{r}` was not given. Use `-h` or `--help` if information on the argument is needed.'

    assert args['output_dir'], 'Output directory not given.'
    if not isinstance(args['output_dir'], Path): args['output_dir'] = Path(args['output_dir'])
    args['output_dir'].mkdir(parents=True, exist_ok=True)
    
    return args

def check_model(args:dict, 
                allowed_models:List[str]=['wavlm-base'],
                allowed_pool:List[str]= ['mean', 'max'],
                allowed_freeze:List[str] = ['all', 'layer', 'none'],
                required:List[str] =['model_type', 'pt_ckpt', 'seed', 'freeze_method', 'pool_method', 'in_features', 'out_features']) -> dict:
    """
    Check if a model config file exists or has been given, load in if it does, and ensure required arguments are there and pass assertions. 

    If the model config file is not an existing model config (as saved in the model class), make sure that it follows the format of those config files. See README for more information. TODO!

    :param args: dictionary of arguments - vars(argparse object)
    :param allowed_models: list of allowed model types
    :param allowed_pool: list of allowed pooling methods
    :param allowed_freeze: list of allowed freeze methods
    :param required: list of required arguments
    :return args: updated args dictionary
    """
    #check for mismatch between existing and not existing model cfg
    existing_cfg = args['output_dir'].rglob('*model_config.json')

    if existing_cfg:
        f = next(existing_cfg)
        precfg = load_config(f)

    #TODO: compare existing model_cfg and given arguments? or just overwrite with existing arguments? 

    if 'model_cfg' in args:
        model_cfg = args.pop('model_cfg')
        base_cfg = model_cfg.pop('base_config')
        clf_cfg = model_cfg.pop('clf_config')

        if 'hf_path' in model_cfg:
            model_cfg['pt_ckpt'] = model_cfg.pop('hf_path')
        if 'ckpt' in clf_cfg:
            clf_cfg['clf_ckpt'] = clf_cfg.pop('ckpt')

        args.update(base_cfg)
        args.update(clf_cfg)
        args.update(model_cfg)

        mt = args['model_type']
        assert mt in allowed_models, f'{mt} is an invalid model type. Choose one of {allowed_models}.'
        fm = args['freeze_method']
        assert fm in allowed_freeze, f'{fm} is an invalid freeze method. Choose one of {allowed_freeze}.'
        pm = args['pool_method']
        assert pm in allowed_pool, f'{pm} is an invalid pooling method. Choose one of {allowed_pool}.'

    else:
        for r in required:
            assert r in args, f'The required argument `{r}` was not given. Use `-h` or `--help` if information on the argument is needed.'

    if 'clf_ckpt' in args:
        if not isinstance(args['clf_ckpt'], Path): args['clf_ckpt'] = Path(args['clf_ckpt'])
    if 'ft_ckpt' in args:
        if not isinstance(args['ft_ckpt'], Path): args['ft_ckpt'] = Path(args['ft_ckpt'])
    
    if args['model_type'] in ['wavlm-base']:
        args['hf_model'] = True
        if not isinstance(args['pt_ckpt'],str): args['pt_ckpt'] = str(args['pt_ckpt'])
    else:
        args['hf_model'] = False

    assert 'freeze_method' in args, 'Freeze method not given. Check command line arguments or model configuration file.'
    if args['freeze_method'] == 'layer': assert 'unfreeze_layers' in args, 'unfreeze_layers must be given if freeze method is `layer`.'

    if args['pool_method'] in ['mean', 'max']: assert 'pool_dim' in args, 'pool_dim not given. Check command line arguments or model configuration file. '
    return args

# PREP ARGUMENTS #
def zip_clf(args:argparse.Namespace) -> dict:
    """
    Zip arguments for classifier into a dictionary
    :param args: argparse Namespace object
    :return clf_args: dict of classifier arguments
    """
    clf_args = {}
    if args.in_features:
        clf_args['in_features'] = args.in_features
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
        model_args = {'model_type':args.model_type, 'hf_path':str(args.pt_ckpt), 'use_featext':args.use_featext,
                    'out_dir':args.output_dir,'freeze_extractor':args.freeze_extractor, 
                    'freeze_method':args.freeze_method, 'pool_method':args.pool_method, 'seed':args.seed}
    if args.sample_rate:
        model_args['target_sample_rate'] = args.sample_rate
    if args.ft_ckpt:
        model_args['ft_ckpt'] = args.ft_ckpt
    if args.pool_dim:
        model_args['pool_dim'] = args.pool_dim
    if args.unfreeze_layers:
        model_args['unfreeze_layers'] = args.unfreeze_layers
    
    else:
        raise NotImplementedError('Only compatible with huggingface models currently.')
    return model_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Parser")
    #OPTIONAL CONFIG FILES
    cfg_args = parser.add_argument_group('cfg', 'configuration file related arguments')
    cfg_args.add_argument('--load_cfg', type=load_config, help='Audio loading configuration json')
    cfg_args.add_argument('--model_cfg', type=load_config, help="Model configuration json")
    #I/O
    io_args = parser.add_argument_group('io', 'file related arguments')
    io_args.add_argument('--output_dir', type=Path, help='Output directory for saving all files.')
    #AUDIO
    audio_args = parser.add_argument_group('audio', 'Audio file related args')
    audio_args.add_argument('--sample_rate', type=int, help='Specify target sample rate. Must line up with model.')
    #BASE MODEL
    model_args = parser.add_argument_group('model', 'model related arguments')
    model_args.add_argument('--model_type', type=str,
                            choices=['wavlm-base'], help='Specify model type')
    model_args.add_argument('--pt_ckpt', type=Path, help='Specify pretrained model path. For hugging face models, this can be the hub name or a local path.')
    model_args.add_argument('--seed', type=int, help='Specify random seed for model initialization.')
    model_args.add_argument('--use_featext', action='store_true', help='Specify whether a separate feature extractor is used.')
    model_args.add_argument('--freeze_extractor', action='store_true', help='Specify whether to freeze the feature extractor.')
    model_args.add_argument('--freeze_method', type=str, choices=['all', 'layer', 'none'], help='Specify what freeze method to use.')
    model_args.add_argument('--unfreeze_layers', nargs="+", help="If freeze_method is `layer`, use this to specify which layers to freeze")
    model_args.add_argument('--ft_ckpt', type=Path, help='Specify finetuned model checkpoint')
    model_args.add_argument('--pool_method', type=str, choices=['mean', 'max', 'attn'], help='Specify pooling method prior to classification head.')
    model_args.add_argument('--pool_dim', type=int, nargs="+", help='Specify the pooling dimension(s).')
    #CLASSIFIER
    clf_args = parser.add_argument_group('classifier', 'classifier related arguments')
    clf_args.add_argument('--in_features', type=int, help="Number of input features to the classification head.")
    clf_args.add_argument('--out_features', type=int, help='Specify number of classification categories (output features).')
    clf_args.add_argument('--nlayers', type=int, help='Specify classification head size.')
    clf_args.add_argument('--activation', type=str, help='Specify type of activation function to use in the classifier.')
    clf_args.add_argument('--clf_ckpt', type=Path, help="Specify classification checkpoint.")

    required_load = ['output_dir']
    allowed_models=['wavlm-base']
    allowed_pool = ['mean', 'max']
    allowed_freeze = ['all', 'layer', 'none']
    required_model =['model_type', 'pt_ckpt', 'seed', 'freeze_method', 'pool_method', 'in_features', 'out_features']
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_l = check_load(args_dict, required_load)
    args_dict_m = check_model(args_dict_l, allowed_models, allowed_pool, allowed_freeze, required_model)
    updated_args = argparse.Namespace(**args_dict_m)

    ma = zip_model(updated_args)
    ca = zip_clf(updated_args)
    ma.update(ca)

    if args.hf_model:
        model = HFModel(**ma) #TODO: add sampling rate check
    else:
        raise NotImplementedError('Currently only compatible with hugging face models.')
    print('pause')
