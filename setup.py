#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

setup(
    name='drqa',
    version='0.1.0',
    description='Reading Wikipedia to Answer Open-Domain Questions',
    long_description=readme,
    license='',
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
    install_requires=[],
)