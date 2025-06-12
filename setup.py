from setuptools import setup, find_packages
from summer25._version import __version__

setup(
    name = 'summer25.py',
    packages = find_packages(),
    author = 'Daniela Wiepert',
    python_requires='>=3.10',
    install_requires=[
        'numpy==2.2.6',
        'pandas==2.3.0',
        'scikit-learn==1.7.0',
        'scipy==1.15.3',
        'timm==1.0.15',
        'torch==2.7.0',
        'torchaudio==2.7.0',
        'torchvision==0.22.0',
        'tqdm==4.67.1',
        'transformers==4.52.4'
    ],
    include_package_data=False,
    version = __version__
)
