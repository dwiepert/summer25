__version__ = '0.5.8'

#change log:
# 0.0.0: initial commit
# 0.0.1: initial code for model classes
# 0.1.0: hugging face model class implemented
# 0.2.0: data split implementation
# 0.2.1: data split testing
# 0.2.2: round 1 of model tests added
# 0.2.3: untested dataset class/waveform loading added
# 0.2.4: untested transforms/io added
# 0.2.5: tested implementations of dataset/transforms/io
# 0.2.6: tested model freezing options
# 0.2.7: untested model pooling/forward pass/padding
# 0.2.8: tests for model pooling/forward pass/padding+attention pooling
# 0.2.9: skeleton for finetune/val/evaluate loops
# 0.2.10: refactor for training, loss option added
# 0.2.11: draft of LoRA, classification head
# 0.2.12: updated saving/loading for HF models (tested)
# 0.3.0: soft prompting/LoRA implemented and tested
# 0.4.0: trainer implemented and tested
# 0.4.1: untested changes to feature extractor/collation/scheduler/optimizer
# 0.4.2: untested changes to pooling
# 0.4.3: tests added (excluding gcs tests)
# 0.4.4: untested gcs upload/download
# 0.4.5: tested gcs upload/download
# 0.5.0: reworked model - need to test run.py
# 0.5.1: mostly debugged & tested version 0.5
# 0.5.2: more debugging/tests
# 0.5.3: fully debugged & tested version 0.5
# 0.5.4: debugging memory issues
# 0.5.5: restructure feature extraction
# 0.5.6: remove data parallel
# 0.5.7: small saving fixes
# 0.5.8: update rank loss