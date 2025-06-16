_MODELS = {  "wavlm-base": {
                  "hf_hub": "microsoft/wavlm-base",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True
            },
            "wavlm-large": {
                  "hf_hub": "microsoft/wavlm-large",
                  "use_featext": True,
                  "in_features": 1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True
            },
            "hubert-base": {
                  "hf_hub": "facebook/hubert-base-ls960",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True
            },
            "hubert-large": {
                  "hf_hub": "facebook/hubert-large-ll60k",
                  "use_featext": True,
                  "in_features":1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True
            },
            "hubert-xlarge": {
                  "hf_hub": "facebook/hubert-xlarge-ll60k",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  "monochannel": True
            },
            "whisper-tiny": {
                  "hf_hub": "openai/whisper-tiny",
                  "use_featext": True,
                  "in_features":384,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True
            },
            "whisper-base": {
                  "hf_hub": "openai/whisper-base",
                  "use_featext": True,
                  "in_features":512,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True
            },
            "whisper-small": {
                  "hf_hub": "openai/whisper-small",
                  "use_featext": True,
                  "in_features":768,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True
            },
            "whisper-medium": {
                  "hf_hub": "openai/whisper-medium",
                  "use_featext": True,
                  "in_features":1024,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True
            },
            "whisper-large": {
                  "hf_hub": "openai/whisper-large",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True
            },
            "whisper-large-v2": {
                  "hf_hub": "openai/whisper-large-v2",
                  "use_featext": True,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30,
                  "monochannel": True
            },
            "test_model": {
                  "use_featext": False,
                  "in_features":1280,
                  "pool_dim": 1,
                  "target_sample_rate": 16000,
                  'clip_length': 30
            }
}

_POOL = ['mean', 'max', 'attn']
_FREEZE = ['all', 'layer', 'none']

_TASKS = ['sentence_repetition', 'word_repetition']

_FEATURES = ['abn_pitch_or_loudness_variability', 'abn_resonance',
       'accelerating_rate', 'audible_false_starts_or_restarts',
       'audible_nasal_emission', 'breathy', 'excess_equal_stress',
       'high_pitch', 'hoarse_harsh', 'hypernasal',
       'inappropriate_silences_or_prolonged_intervals', 'intelligibility',
       'irregular_artic_breakdowns', 'language', 'loudness_decay', 'low_pitch',
       'monopitch_monoloudness', 'other_cognitive_communication_skills',
       'prolonged_sounds', 'rapid_rate', 'reduced_syllables_per_breath_group',
       'repeated_sounds', 'slow_rate', 'sound_additions', 'sound_distortions',
       'sound_omissions', 'sound_sequencing_errors', 'sound_substitutions',
       'strained', 'stridor', 'syllable_segmentation',
       'telescoping_of_syllables', 'verbal_asides', 'vocal_noises',
       'voice_interruptions', 'word_or_phrase_repetitions']

