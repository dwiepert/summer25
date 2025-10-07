import json 
import tempfile
from pathlib import Path
from typing import Union
from summer25.io import search_gcs, download_to_local 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
import pandas as pd
import re

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
        sub_dirs = list(set([f.parent for f in files if str(f.parent.parent) == str(parent_directory)]))
    else:
        sub_dirs = [d for d in parent_directory.iterdir() if (str(d.parent) == str(parent_directory) and d.is_dir())]

    data = {}
    for s in sub_dirs:
        data[s] = get_data(s, bucket)
    
    return data

def check_config_v1(data):
    """
    """
    incorrect_d = {}
    for d in data:
        vals = []
        incorrect_vals = {}
        split_d = str(d.name).split("_")
        if split_d[13] == '':
            check = {'model_type':split_d[0], 'seed':int(re.findall(r'\d+', split_d[1])[0]), 'freeze_method':split_d[2], 'pool_method':split_d[3], 'finetune_method':split_d[4],
                    'clf_type': split_d[6], 'in_features':int(".".join(re.findall(r'\d+', split_d[7]))), 'out_features':int(".".join(re.findall(r'\d+', split_d[8]))), 'activation': split_d[9],
                    'nlayers': int(".".join(re.findall(r'\d+', split_d[10]))), 'optim_type':split_d[12], 'learning_rate':float(".".join(re.findall(r'\d+', split_d[14]))),
                    'tf_learning_rate':float("e-".join(re.findall(r'\d+', split_d[15]))), 'loss_type':split_d[16],'rating_threshold':float(".".join(re.findall(r'\d+', split_d[17]))), 'margin':float(".".join(re.findall(r'\d+', split_d[18]))),
                    'bce_weight':float(".".join(re.findall(r'\d+', split_d[19]))),'scheduler_type':split_d[20], 'train_len':int(".".join(re.findall(r'\d+', split_d[21]))), 'early_stop':True, 'batch_size':int(re.findall(r'\d+', split_d[23])[0]), 'gradient_accumulation_steps':int(re.findall(r'\d+', split_d[24])[0]),
                    'epochs':int(re.findall(r'\d+', split_d[25])[0])}
        elif split_d[4] == 'lora' or split_d[4] == 'soft-prompt':
            check = {'model_type':split_d[0], 'seed':int(re.findall(r'\d+', split_d[1])[0]), 'freeze_method':split_d[2], 'pool_method':split_d[3], 'finetune_method':split_d[4],
                    'clf_type': split_d[6], 'in_features':int(".".join(re.findall(r'\d+', split_d[7]))), 'out_features':int(".".join(re.findall(r'\d+', split_d[8]))), 'activation': split_d[9],
                    'nlayers': int(".".join(re.findall(r'\d+', split_d[10]))), 'optim_type':split_d[11], 'loss_type':split_d[12], 'learning_rate':float(".".join(re.findall(r'\d+', split_d[13]))),
                    'tf_learning_rate':float("e-".join(re.findall(r'\d+', split_d[14]))), 'rating_threshold':float(".".join(re.findall(r'\d+', split_d[16]))), 'margin':float(".".join(re.findall(r'\d+', split_d[17]))),
                                    'bce_weight':float(".".join(re.findall(r'\d+', split_d[18]))), 'early_stop':True, 'batch_size':int(re.findall(r'\d+', split_d[20])[0]), 'gradient_accumulation_steps':int(re.findall(r'\d+', split_d[21])[0]),
                    'epochs':int(re.findall(r'\d+', split_d[22])[0])}
        else:
            check = {'model_type':split_d[0], 'seed':int(re.findall(r'\d+', split_d[1])[0]), 'freeze_method':split_d[2], 'pool_method':split_d[3],
                    'clf_type': split_d[5], 'in_features':int(".".join(re.findall(r'\d+', split_d[6]))), 'out_features':int(".".join(re.findall(r'\d+', split_d[7]))), 'activation': split_d[8],
                    'nlayers': int(".".join(re.findall(r'\d+', split_d[9]))), 'optim_type':split_d[10], 'loss_type':split_d[11], 'learning_rate':float(".".join(re.findall(r'\d+', split_d[12]))),
                    'tf_learning_rate': float("e-".join(re.findall(r'\d+', split_d[13]))), 'rating_threshold':float(".".join(re.findall(r'\d+', split_d[15]))), 'margin':float(".".join(re.findall(r'\d+', split_d[16]))),
                                    'bce_weight':float(".".join(re.findall(r'\d+', split_d[17]))), 'early_stop':True, 'batch_size':int(re.findall(r'\d+', split_d[19])[0]), 'gradient_accumulation_steps':int(re.findall(r'\d+', split_d[20])[0]),
                    'epochs':int(re.findall(r'\d+', split_d[21])[0])}


        model = data[d]['model_config.json']
        train = data[d]['train_config.json']

        for m in model:
            if m in check:
                vals.append(model[m] == check[m])
                spec = check[m]
                act = model[m]
                if model[m] != check[m]:
                    print(f'specified: {spec} vs. actual: {act}')
                    incorrect_vals[m] = {'specified': check[m], 'actual': model[m]}

        for m in train:
            if m in check:
                vals.append(train[m] == check[m])
                if train[m] != check[m]:
                    spec = check[m]
                    act = train[m]
                    print(f'specified: {spec} vs. actual: {act}')
                    incorrect_vals[m] = {'specified': check[m], 'actual': train[m]}
        if not all(vals):
            incorrect_d[str(d)] = incorrect_vals

    return incorrect_d


def create_data_csvs(parent_directory,  bucket, savedir):
    if not isinstance(savedir, Path): savedir = Path(savedir)
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)
    data = extract_data_from_parentdir(parent_directory, bucket)

    incorrect_d = check_config_v1(data)

    with open(str(savedir/'incorrect.json'), 'w') as f:
        json.dump(incorrect_d, f, indent=4)

    #assert not check_config_v1(data)

    metadata_dict = {'file_path':[]}
    training_dict = {'file_path':[]}
    eval_dict = {'file_path':[]}
    outputs = {'file_path':[]}
    for k in data:
        if k in incorrect_d:
            continue 

        model_path = str(k)
        model_data = data[k]
        model_md = model_data['model_config.json']
        
        temp = metadata_dict['file_path']
        temp.append(model_path)
        metadata_dict['file_path'] = temp

        for m in model_md:
            if m not in metadata_dict:
                metadata_dict[m] = [model_md[m]]
            else:
                temp = metadata_dict[m]
                temp.append(model_md[m])
                metadata_dict[m] = temp 
        
        train_md = model_data['train_config.json']
        for m in train_md:
            if m not in metadata_dict:
                metadata_dict[m] = [train_md[m]]
            else:
                temp = metadata_dict[m]
                temp.append(train_md[m])
                metadata_dict[m] = temp 

        train_log = model_data['train_log.json']
        length = None
        for d in train_log:
            item = train_log[d]
            if isinstance(item, list) and length is None:
                length = len(item)
            elif not isinstance(item, list):
                new_item = [item] * length 
                item = new_item
            else:
                assert length == len(item)
            if d not in training_dict:
                training_dict[d] = item
            else:
                temp = training_dict[d]
                temp.extend(item)
                training_dict[d] = temp
        
        epochs = list(range(length)) 
        if 'epoch' not in training_dict:
            training_dict['epoch'] = epochs
        else:
            temp = training_dict['epoch']
            temp.extend(epochs)
            training_dict['epoch'] = temp

        temp = training_dict['file_path']
        temp.extend([model_path]*length)
        training_dict['file_path'] = temp 

        evaluation = model_data['evaluation.json']

        # for binary
        target_features = evaluation['target_features']
        probabilities = evaluation.pop('probabilities')
        binary_targets = evaluation.pop('binary_targets')

        p_df = pd.DataFrame(probabilities, columns=target_features)
        p_df['type'] = 'probabilities'
        bt_df = pd.DataFrame(binary_targets, columns=target_features)
        bt_df['type'] = 'binary_targets'

        binary = pd.concat([p_df, bt_df])
        binary.to_csv(savedir / 'binary_outputs.csv', index=False)

        # direct outputs/targets
        outputs = evaluation.pop('outputs')
        targets = evaluation.pop('targets')

        o_df = pd.DataFrame(outputs, columns = target_features )
        t_df = pd.DataFrame(targets, columns=target_features)
        o_df['type'] = 'raw_outputs'
        t_df['type'] = 'raw_targets'
        
        out = pd.concat([o_df, t_df])
        out.to_csv(savedir/'raw_outputs.csv', index=False)
       
        
        #target_features = evaluation.pop('target_features')

        eval_length = len(probabilities)
        feats = None
        temp_dict = {}
        for d in evaluation:
            if d not in ['target_features','macro_auc', 'overall_acc', 'bacc_per_feature', 'acc_per_feature', 'auc_per_feature', 'test_loss', 'avg_test_loss', 'best_epoch']:
                continue

            item = evaluation[d]

            if not isinstance(item, list):
                to_add = [item]*len(target_features)
            elif len(item) == len(target_features):
                to_add = item
            else:
                print('pause')

            if d not in eval_dict:
                eval_dict[d] = to_add
            else:
                temp = eval_dict[d]
                temp.extend(to_add)
                eval_dict[d] = temp
            
            # else:
            #     if len(item) == len(target_features):
            #         if d not in eval_
            #     if len(item) != len(target_features):
            #         if eval_length is None:
            #             eval_length = len(item)
            #         else:
            #             assert eval_length == len(item)
                    
            #         new_list = []
            #         features = []
            #         for i in range(len(item)):
            #             for j in range(len(target_features)):
            #                 new_list.append(item[i][j])
            #                 features.append(target_features[j])
                    
            #         if ex_feats is None: ex_feats=features
            #         assert len(new_list) == len(features)

            #         if d not in eval_dict:
            #             eval_dict[d] = new_list
            #             #eval_dict['target_features'] = features
            #         else:
            #             temp_list = eval_dict[d]
            #             temp_list.extend(new_list)
            #             eval_dict[d] = temp_list
            #            # temp_feats = eval_dict['target_features']
            #            # temp_feats.extend(features)

            #             #assert len(temp_feats) == len(temp_list)
            #            # eval_dict['target_features'] = temp_feats
            #     else:
            #         if eval_length is None:
            #             temp_dict[d] = [item]
                    
            #         else: 
            #             new_item = item * (eval_length)
            #             features = target_features * eval_length
                    
            #         if ex_feats2 is None:
            #             ex_feats2 = features
            #         assert len(new_item) == len(features)


            #         list_len = len(new_item)
            #         if d not in eval_dict:
            #             eval_dict[d] = new_item
            #         else:
            #             temp_list = eval_dict[d]
            #             temp_list.extend(new_item)
            #             list_len = len(temp_list)
            #             eval_dict[d] = temp_list

                    # if 'target_features' not in eval_dict:
                    #     eval_dict['target_features'] = features
                    # else:
                    #     temp_feats = eval_dict['target_features']
                    #     temp_feats.extend(features)
                    #     assert len(temp_feats) == list_len
                    #     eval_dict['target_features'] = temp_feats
                    
            # else:
            #     new_item = [item] * (eval_length*len(target_features))
            #     features = target_features * eval_length
                
            #     if ex_feats3 is None:
            #         ex_feats3 = features
            #     assert len(new_item) == len(features)
            #     list_len = len(new_item)
            #     if d not in eval_dict:
            #         eval_dict[d] = new_item
            #     else:
            #         temp_list = eval_dict[d]
            #         temp_list.extend(new_item)
            #         list_len = len(temp_list)
            #         eval_dict[d] = temp_list
                    
                # if 'target_features' not in eval_dict:
                #     eval_dict['target_features'] = features
                # else: 
                #     temp_feats = eval_dict['target_features']
                #     temp_feats.extend(features)
                #     assert len(temp_feats) == list_len
                #     eval_dict['target_features'] = temp_feats
        
        


        temp = eval_dict['file_path']
        temp.extend([model_path]*len(target_features))
        eval_dict['file_path'] = temp 

    metadata_df = pd.DataFrame(metadata_dict)
    metadata_df.to_csv(savedir / 'metadata.csv', index=False)
    training_df = pd.DataFrame(training_dict)
    training_df.to_csv(savedir/'training.csv', index=False)
    
    eval_df = pd.DataFrame(eval_dict)
    eval_df.to_csv(savedir/'eval.csv', index=False)

def plot_training_loss(data, save_dir):
    """
    """    
    #colors = list(mcolors.TABLEAU_COLORS)
    #r_colors = random.sample(colors, len(data))
    i = 0
    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr= train_md['tf_learning_rate']
        lr = train_md['learning_rate']


        plt.plot(output['train_log.json']['avg_train_loss'], label= f'Training loss - tf_lr{tflr}; lr{lr}')#, color=r_colors[i])
        plt.plot(output['train_log.json']['avg_val_loss'], label= f'Val loss - tf_lr{tflr}; lr{lr}', linestyle='--')#, color=r_colors[i])
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


        plt.plot(output['train_log.json']['avg_train_loss'], label= f'Training loss - tf_lr{tflr}; lr{lr}')#, color=r_colors[i])
        i += 1

    plt.title('Average loss during training')
    plt.xlabel('Avg. loss')
    plt.ylabel('Epochs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(str(Path(save_dir) / 'training_loss.png'), bbox_inches='tight', dpi=300)
    plt.close()

    i=0
    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr= train_md['tf_learning_rate']
        lr = train_md['learning_rate']


        plt.plot(output['train_log.json']['avg_val_loss'], label= f'Validation loss - tf_lr{tflr}; lr{lr}', linestyle='--')#, color=r_colors[i])
        i += 1

    plt.title('Average validation loss')
    plt.xlabel('Avg. loss')
    plt.ylabel('Epochs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(str(Path(save_dir) / 'validation_loss.png'), bbox_inches='tight', dpi=300)
    plt.close()

    i = 0
    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr= train_md['tf_learning_rate']
        lr = train_md['learning_rate']


        plt.plot(output['train_log.json']['avg_train_loss'], label= f'Training loss - tf_lr{tflr}; lr{lr}', color="black")
        plt.plot(output['train_log.json']['avg_val_loss'], label= f'Val loss - tf_lr{tflr}; lr{lr}', linestyle='--', color="red")
        plt.title('Average loss during training')
        plt.xlabel('Avg. loss')
        plt.ylabel('Epochs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(str(Path(save_dir) / f'train_val_loss_tflr{tflr}_lr{lr}.png'), bbox_inches='tight', dpi=300)
        plt.close()
        i += 1

def plot_auc(data, save_dir):
    """
    """
    if not isinstance(save_dir, Path): save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok = True)
    colors = list(mcolors.TABLEAU_COLORS)
    r_colors = random.sample(colors, len(data))
    categories = []
    features = {}
    feats = []
    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr= train_md['tf_learning_rate']
        lr = train_md['learning_rate']
        categories.append(f'tflr{tflr}_lr{lr}')
        eval = output['evaluation.json']
        feat_list = eval['target_features']
        for i in range(len(feat_list)):
            k = feat_list[i]
            auc = eval['auc_per_feature'][i]
            if k == 'inappropriate_silences_or_prolonged_intervals':
                name = 'ispi'
            if k == 'monopitch_monoloudness':
                name = 'mm'
            if k == 'hoarse_harsh':
                name = 'hh'
            if k == 'slow_rate':
                name = 'sr'
            if k == 'sound_distortions':
                name = 'sd'
            if name not in feats:
                feats.append(name)

            if not auc:
                auc = 0
            if name in features:
                temp = features[name]
                temp.append(auc)
                features[name] = temp
            else:
                features[name] = [auc]
    

    #AUC across learning rates
    x = np.arange(len(categories))
    bar_width = 0.15

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in features.items():
        offset = bar_width * multiplier
        rects = ax.bar(x + offset, measurement, bar_width, label=attribute)
        #ax.bar_label(rects, padding=3)
        multiplier += 1
        
    ax.set_ylabel('AUC')
    ax.set_title('AUC across learning rates')
    ax.set_xticks(x + bar_width, categories)
    ax.set_xticklabels(categories, rotation=90)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    #plt.xticks(rotation=90)
    plt.savefig(str(Path(save_dir) / f'auc_lr.png'), bbox_inches='tight', dpi=300)
    plt.close()

    #AUC per feature
    multiplier = 0
    x = np.arange(len(feats))
    bar_width = 0.1
    fig, ax = plt.subplots(layout='constrained')

    for i in range(len(categories)):
        c = categories[i]
        aucs = []
        for f in features:
            aucs.append(features[f][i])
        
        offset = bar_width * multiplier
        rects = ax.bar(x + offset, aucs, bar_width, label=c)
        multiplier += 1

    ax.set_ylabel('AUC')
    ax.set_title(f'AUC across features')
    ax.set_xticks(x+bar_width, feats)
    #ax.set_xticklabels(feats, rotation=90)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    #plt.xticks(rotation=90)
    plt.savefig(str(Path(save_dir) / 'auc_features.png'), bbox_inches='tight', dpi=300)
    plt.close()

    multiplier = 0
    x = np.arange(len(feats))

    for i in range(len(categories)):
        bar_width = 0.5
        fig, ax = plt.subplots(layout='constrained')
        c = categories[i]
        aucs = []
        for f in features:
            aucs.append(features[f][i])
        
        rects = ax.bar(x, aucs, bar_width, label=c)

        ax.set_ylabel('AUC')
        ax.set_title(f'AUC across features: {c}')
        ax.set_xticks(x, feats)
        #ax.set_xticklabels(feats, rotation=90)
        #ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 1)
        #plt.xticks(rotation=90)
        plt.savefig(str(Path(save_dir) / f'auc_features_{c}.png'), bbox_inches='tight', dpi=300)
        plt.close()
        

    
    
    
def plot_accuracy(data, save_dir):
    """
    """
    if not isinstance(save_dir, Path): save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok = True)
    colors = list(mcolors.TABLEAU_COLORS)
    r_colors = random.sample(colors, len(data))
    categories = []  # X positions for categories
    bar_width = 0.25

    features = {}
    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr= train_md['tf_learning_rate']
        lr = train_md['learning_rate']
        categories.append(f'tflr{tflr}_lr{lr}')
        eval = output['evaluation.json']
        feat_list = eval['target_features']
        bacc_per_feature = eval['bacc_per_feature']
        acc_per_feature = eval['acc_per_feature']
        for i in range(len(feat_list)):
            k= feat_list[i]
            if k in features:
                temp_bacc = features[k]['bacc']
                temp_bacc.append(bacc_per_feature[i])
                temp_acc = features[k]['acc']
                temp_acc.append(acc_per_feature[i])
                features[k]['bacc'] = temp_bacc
                features[k]['acc'] = temp_acc
            else:
                features[k] = {'bacc':[bacc_per_feature[i]], 'acc':[acc_per_feature[i]]}
        
    #i = 0
    x = np.arange(len(categories))
    for f in features:
        feat_dict = features[f]
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained')
        for attribute, measurement in feat_dict.items():
            offset = bar_width * multiplier
            rects = ax.bar(x + offset, measurement, bar_width, label=attribute)
            #ax.bar_label(rects, padding=3)
            multiplier += 1
        
        ax.set_ylabel('Accuracy score')
        ax.set_title('Accuracy across learning rates')
        ax.set_xticks(x + bar_width, categories)
        ax.set_xticklabels(categories, rotation=90)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 1)
        #plt.xticks(rotation=90)
        plt.savefig(str(Path(save_dir) / f'accuracy_{f}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        #print('pause')

    #inverse plot
    for i in range(len(categories)):
        fig, ax = plt.subplots(layout='constrained')
        multiplier = 0
        bacc = []
        acc = []
        feats = []
        for f in features:
            if f == 'inappropriate_silences_or_prolonged_intervals':
                feats.append('ispi')
            if f == 'monopitch_monoloudness':
                feats.append('mm')
            if f == 'hoarse_harsh':
                feats.append('hh')
            if f == 'slow_rate':
                feats.append('sr')
            if f == 'sound_distortions':
                feats.append('sd')

            feat_dict = features[f]
            bacc.append(feat_dict['bacc'][i])
            acc.append(feat_dict['acc'][i])
        x = np.arange(len(feats))
        offset = bar_width * multiplier
        rects = ax.bar(x + offset, bacc, bar_width, label="bacc")
        multiplier += 1

        offset = bar_width * multiplier
        rects = ax.bar(x + offset, acc, bar_width, label="acc")

        ax.set_ylabel('Accuracy score')
        ax.set_title(f'Accuracy across features: {categories[i]}')
        ax.set_xticks(x + bar_width, feats)
        ax.set_xticklabels(feats, rotation=90)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 1)
        plt.savefig(str(Path(save_dir) / f'accuracy_{categories[i]}.png'), dpi=300)
        plt.close()


def get_best_epoch(data):
    tflr = []
    lr = []
    best = []

    for f in data:
        output = data[f]
        train_md = output['train_config.json']
        tflr.append(train_md['tf_learning_rate'])
        lr.append(train_md['learning_rate'])
        best.append(output['best*'])
    
    df = pd.DataFrame({'tflr':tflr, 'lr':lr, 'best_epoch':best})
    print(df)
    return df