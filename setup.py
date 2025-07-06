#!/usr/bin/env python3
# coding: utf-8

"""
setup.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/)

"""

from setuptools import setup, find_packages

setup(
name='nlqk',
version='0.0.3',
author='Damir Cavar, James Bryan Graves, Billy G. Dickson, Koushik Reddy Parukola, Shane A. Sparks',
author_email='dcavar@iu.edu, gravjabr@iu.edu, dicksonb@iu.edu, koparu@iu.edu, sparkssh@iu.edu',
description='A Quantum AI and Natural Language Processing (Q-NLP) package',
packages=find_packages(),
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
install_requires=[
	'numpy',
	'pandas',
	'scipy',
	'requests',
	'cupy-cuda12x'
],
python_requires='>=3.9',
)
