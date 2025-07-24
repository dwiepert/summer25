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
* freezing/finetuning: (if you are going to add parameters to model - could add to classifier level, limited compute/limited data - where to add it? Add one more classifier layer and see what happens? More data you have, the more parameters you can reasonably optimize - we're always low data, so what's the best way to limit the data.)
    * all 
    * exclude-last 
    * half
    * required-only
    * LoRA
    * soft-prompting
    * add one layer to classifier
* pooling
    * mean
    * max
    * attention
* MODELS:
    * wavlm-large
    * hubert-large
    * whisper-medium
* scheduler: TODO
    * cosine, #skip warmup
    * do by epoch not training step bc it's a small thought


## QUESTIONS/RESEARCH
*  5 MOST COMMON FEATURES IN SENTENCE REPETITION - REDO WITH NEW DC EVENTUALLY TODO:
        * hoarse_harsh: 452
        * slow_rate: 402
        * sound_distortions: 349
        * monopitch_monoloudness: 341
        * inappropriate_silences_or_prolonged_intervals: 264
        ```
        data = pd.read_csv('CSV')
        data_feats = data[_FEATURES]
        freq = (data_feats > 1).sum()
        print(freq.sort_values()[-5:])
        ```



## ACTIVE DEBUGGING/TASKS
* #TODO: check what milestone should be based on warmup epoch? - make sure it's aligned properly (possible test to check?)* not important
* max pooling done right?
* fix run.py for proper inputs
* LATER LATER LATER: make code more concise?

## All TODO
* ~~Make seeded_split compatible with gcs~~ 
* ~~load from existing configuration~~
* ~~load huggingface models~~ 3
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

