# Summer 2025 Project
REQUIREMENTS:
    audio directory format
    metadata expectations

CONFIG FILES:
    - run.py only contains some parameters for audio
    - can specify more exact specifics for WavDataset transforms by adding to a config file. TODO

## QUESTIONS/RESEARCH
* get whisper final hidden state - make it flexible?, freeze it vs. unfreeze it? 
* exclude some of the tokens - see if things are coming out in different sizes? need to figure out Padding tokens and determine which ones to pool over - determine padding, move the feature extractor to do the padding for me - get attention mask out from it, one collate function
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
* determine what classifier build to use with Rankings? It's not going to work the same as BCE loss...not entirely sure how to predict rank - multiclass/multicategory? ask leland what he did
* tf_lr 1e-6 bigger adjustments to classifier. 


## ACTIVE DEBUGGING/TASKS
* CHECK ON POOLING W ATTENTION MASK USING WHISPER? UNCERTAIN WHAT THE SHAPE OF THAT IS AND NEED TO KNOW
* make seeded_split/model loading/model saving compatible with gcs
* TEST SUITE
    * io/transforms - GCS test google cloud things????? mark to run only sometimes
    * re-test new schedulers
    * retest train params (new options)
    * test multiple learning rate options (default tf_lr for unfrozen one should be closer to 1e-6)
    * test new feature extractor location
    * test that all the PEFT models run through... and can be updated
    * TEST ATTENTION MASK STUFF


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
    * ignore padding
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

