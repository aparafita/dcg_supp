#!/usr/bin/env python

from setuptools import setup

version = '0.1.1'

long_description = """
# dcg

This project implements Deep Causal Graphs and contains
several implementations of its Deep Causal Unit (DCU), 
from distributional DCUs to Normalizing Causal Flows.

This technique can model Structural Causal Models
with complex data distributions, and be used to perform
causal inference from interventions to counterfactuals.

For more information, 
please look at our [Github page](https://github.com/aparafita/dcg).
"""

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name='dcg',
    packages=['dcg', 'dcg.distributional'],
    version=version,
    license='MIT',
    description='Deep Causal Graphs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Álvaro Parafita',
    author_email='parafita.alvaro@gmail.com',
    url='https://github.com/aparafita/dcg',
    download_url=f'https://github.com/aparafita/dcg/archive/v{version}.tar.gz',
    keywords=[
        'deep causal graphs', 'causality', 'intervention',
        'counterfactual', 'structural causal model', 'SCM', 'DCG', 
        'density', 'estimation', 
        'sampling', 'probability', 'distribution'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
    ],
    install_requires=install_requires,
    include_package_data=True,
)
