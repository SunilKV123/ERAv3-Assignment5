from setuptools import setup, find_packages

setup(
    name="mnist_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'pytest'
    ]
) 