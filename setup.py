from setuptools import setup, find_packages
from summer25._version import __version__

setup(
    name = 'summer25.py',
    packages = find_packages(),
    author = 'Daniela Wiepert',
    python_requires='>=3.10',
    install_requires=[
        "accelerate==1.8.1",
        "audiomentations==0.41.0",
        "gcsfs==2025.7.0",
        "google-cloud-storage==3.2.0",
        "numpy<2",
        "scikit-learn==1.7.0",
        "scipy==1.15.3",
        "librosa==0.10.2.post1",
        "matplotlib==3.10.6",
        "pandas==2.3.0",
        "huggingface-hub==0.32.4",
        "peft==0.15.2",
        "pytest==8.4.0",
        "tqdm==4.67.1",
        "transformers==4.52.4",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "torchvision==0.22.0"
    ],
    include_package_data=False,
    version = __version__
)
