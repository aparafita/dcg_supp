#!/usr/bin/env python

from setuptools import setup

version = '0.0.2'

long_description = """
# torch_misc

Miscellanious utilities for PyTorch experiments.
"""

with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name='torch-misc',
    packages=['torch_misc'],
    version=version,
    license='MIT',
    description='Utilities for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Álvaro Parafita',
    author_email='parafita.alvaro@gmail.com',
    url='https://github.com/aparafita/torch-misc',
    #download_url=f'https://github.com/aparafita/flow/archive/v{version}.tar.gz',
    keywords=[
        'torch', 'training', 'utilities'
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