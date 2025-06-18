# Summer 2025 Project
REQUIREMENTS:
    audio directory format
    metadata expectations

## QUESTIONS
* safe for features to be public? tasks to be public?
* What’s 1.5 vs 2 for our purposes in features: Binarize - but 1.5 to 0, 2> will be 1 (MAKE SURE TO GET BOTH >2 and also regular 1,2,3, etc). Leland - for one of the models that he trained, he came up with a combo loss that is BCE and w/in each batch, used more finegrained ratings to get the rank of the samples in that batch correctly (mildest to most severe, can learn mild to severe). Get splits that aren't yet binarized, add transform to binarize. 
* why pooled feature annotations?^^ how do we pool this over annotator. Max abnormality (whichever is highest)***
* which features to use for debugging
* is the data context in the example up to date - data context is the big query last, ask leland to double check which data context and for snippet of combo loss so that I have it. 
* keep_extractor for models without feature extractor!
* what are our metrics of interest, just accuracy? balanced accuracy? anything else'
* figure out the feature extractor for WavLM - don't unfreeze it? add option to unfreeze? Double checkrequires.grad =False or True for model as in. 
* Always freeze whisper decoder, keep
* freeze - most common is it to do last layer or unfreeze the whole thing, half might also be interesting (early transformations are more general)
* LOSS FUNCTION - BCE 
* assume CNN that goes wavform to tokens is fixed - SHOUKLD ALWAYS FREEZE
* key and query in LoRA. common things - figure out what the standard things are for HuBERT/WavLM/Whisper. 
* don't pool over Padding tokens!!!!!!! in any pooling!!! how do we ensure that? what does that mean. 

## ACTIVE DEBUGGING/TASKS
* TEST SUITE - ALL ASSERTIONS!
    * test google cloud things????? mark to run only sometimes, check out pytest slow as well to skip some stupp
    * ~~split~~
    * ~~hugging face models~~
    * ~~classifier~~
    * ~~base model~~
    * base dataset
    * io
    * transforms
    * freeze/pool/forward
    * loops/metrics
* unfreeze layers
    * int vs. string
    * encoder only? whisper-specific
    * thoughts on extractor being frozen even during all? (keep_extractor)

## All TODO
* ~~load from existing configuration~~
* ~~load huggingface models~~ 
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper (make sure it recognizes whisper separately)~~
    * ~~Check hugging face hub download and load - + check delete_post (change to delete_download)~~
    * ~~Check local checkpoint load~~
    * ~~confirm configuration is saving properly~~
    * ~~tests~~
* random weight initialization - check seed for classifier and base model
    * tests
* processor - tokenizer needed?
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~
* freeze by layers (decide layer configurations) for -  assert int or str
    * ~~wavlm~~
    * ~~hubert~~
    * ~~whisper~~ - should whisper decoder always be frozen since we aren't using it at all?^^
    * set module names - can then just give layers? determine which modules to freeze
    * keep extractor
        * hubert???
        * wavlm
    * tests
* Classification head 
    * decide basic configurations/layer options
    * decide test features vs. all features? different groupings? what is the plan here
* Add LoRA option
* Add soft prompting option 
* ~~Flexibly create data splits - each seed has a different split?~~ 
* Load data into a dataset
    * transform to 16000 if not done
    * optional trim (FOR WHISPER!!)
    * convert to tensor
    * normalize data?
    * check batching collate_fn
    * use model specific processors????
* pooling strategies
    * determine what pool dimension you need
    * mean
    * max
    * attention
    * tests
* test run forward pass of models
    * do we need model-specific feature processors????
    * wavlm
    * hubert
    * whisper^^
    * tests (include classifier only)
* CONFIRM METRICS
* model loops
    * training + logging options - WHAT IS THE BEST LOSS? OPTIMIZER? ETC.
    * validation + logging options
    * save best model (see other code for tracking the best model) + final model? 
    * testing - what testing do we want to do 
* ~~check finetuned checkpoint loading~~
    * ~~base model~~
    * ~~classification head~~
* visualizations 
    * training vs. validation loss
    * weights? 
    * any kind of attention heads during pooling?
    * outputs


