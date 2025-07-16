# Summer 2025 Project
REQUIREMENTS:
    audio directory format
    metadata expectations

CONFIG FILES:
    - run.py only contains some parameters for audio
    - can specify more exact specifics for WavDataset transforms by adding to a config file. TODO

## TRAINING PARAMS
* FEATURES: hoarse_harsh, slow_rate, sound_distortions, monopitch_monoloudness, 'inappropriate_silences_or_prolonged_intervals'
* batch_size: 16
* learning_rate: 0.001, 0.01, 0.0001
* tf_learning_rate: 1e-6, 1e-5, 1e-4
* loss: rank
    * bce_weight: 0, 0.25, 0.5, 1
* number of classifier layers: TODO
* freezing/finetuning:
    * all 
    * exclude-last 
    * half
    * required-only
    * LoRA
    * soft-prompting
* pooling
    * mean
    * max
    * attention
* MODELS:
    * wavlm-large
    * hubert-large
    * whisper-medium
* scheduler: TODO
    * warmup-cosine, # warmup epochs


## QUESTIONS/RESEARCH
*  5 MOST COMMON FEATURES IN SENTENCE REPETITION - REDO WITH NEW DC EVENTUALLY TODO:
        * hoarse_harsh: 430
        * slow_rate: 355
        * sound_distortions: 312
        * monopitch_monoloudness: 299
        * inappropriate_silences_or_prolonged_intervals: 251
        ```
        data = pd.read_csv('CSV')
        data_feats = data[_FEATURES]
        freq = (data_feats > 1).sum()
        freq.sort_values()[-5]
        ```
* determine what classifier build to use with Rankings? It's not going to work the same as BCE loss...not entirely sure how to predict rank - multiclass/multicategory? ask leland what he did? ask what had been done before? - if you are going to add parameters to model - could add to classifier level, limited compute/limited data - where to add it? Add one more classifier layer and see what happens? More data you have, the more parameters you can reasonably optimize - we're always low data, so what's the best way to limit the data. 
* check 2024/2025 papers for audio classification to see what learning rates/schedulers they use - just choose a scheduler/learning rate - probably do a mini search - 


## ACTIVE DEBUGGING/TASKS
* get new speech dataset loaded
* IMPLEMENT BEATS MODEL
* make seeded_split/model loading/model saving compatible with gcs
* #TODO: check what milestone should be based on warmup epoch?
* max pooling done right?
* TEST SUITE
    * io/transforms - GCS test google cloud things????? mark to run only sometimes
    * check when loading trained model that the weights are the same!!! output same
        * peft model - train for a handful of epochs, check when reloaded that it works as expected

## All TODO
* BEATs model
* Make seeded_split compatible with gcs + run
* ~~load from existing configuration~~
* ~~load huggingface models~~ 
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper (make sure it recognizes whisper separately)~~
    * ~~Check hugging face hub download and load - + check delete_post (change to delete_download)~~
    * ~~Check local checkpoint load~~
    * ~~confirm configuration is saving properly~~
    * ~~tests~~
* ~~processor - tokenizer needed?~~
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
* ~~freeze by layers (decide layer configurations) for -  assert int or str~~
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
    * ~~set module names - can then just give layers? determine which modules to freeze~~
    * ~~tests~~
* ~~Classification head~~
    * ~~decide basic configurations/layer options~~
    * ~~random weight initialization for classifier - check seed~~
    * ~~tests~~
* ~~Add LoRA option~~
    * ~~tests~~
* ~~Add soft prompting option~~
    * ~~test~~
* ~~Flexibly create data splits - each seed has a different split?~~ 
* ~~Load data into a dataset~~
    * ~~transform to 16000 if not done~~
    * ~~optional trim (FOR WHISPER!!)~~
    * ~~convert to tensor~~
    * ~~normalize data?^^~~
    * ~~check batching collate_fn~~
    * ~~use model specific processors????~~
* ~~pooling strategies~~
    * ~~mean~~
    * ~~max~~
    * ~~attention~~
    * ~~tests~~
    * ~~ignore padding~~
* ~~test run forward pass of models~~
    * ~~do we need modsel-specific feature processors????~~
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
    * ~~tests (include classifier only)~~
* ~~model loops~~
    * ~~training + logging options~~
    * ~~validation + logging options~~
    * ~~save best model (see other code for tracking the best model) + final model?~~ 
    * ~~testing - what testing do we want to do ~~
* ~~check finetuned checkpoint loading~~
    * ~~base model~~
    * ~~classification head~~
* visualizations 
    * training vs. validation loss
    * weights? 
    * any kind of attention heads during pooling?
    * outputs

