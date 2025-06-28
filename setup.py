import os
from setuptools import setup, find_packages

def parse_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='liftover2d_pre_alpha',
    version='0.0.1',
    packages=find_packages(),
    install_requires=parse_requirements(),
)