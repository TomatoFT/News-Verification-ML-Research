import subprocess

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


# Custom command to download models
class DownloadModelsCommand(install):
    def run(self):
        # Download models and tokenizers
        subprocess.run(["python", "-m", "actions.download_models."])

# Custom command for development mode
class DevelopCommand(develop):
    def run(self):
        # Download models and tokenizers
        subprocess.run(["python", "-m", "actions.download_models"])
        # Continue with the regular development installation
        develop.run(self)

setup(
    name='UIT-News-Verification',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1',
        'transformers==4.35.2',
        'sacremoses==0.1.1',
        'sentencepiece==0.1.99',
        'pyspark==3.5.0',
        'tqdm==4.66.1',
    ],
    cmdclass={
        'install': DownloadModelsCommand,
        'develop': DevelopCommand,
    },
)
