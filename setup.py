from setuptools import setup, find_packages

setup(
name='nlqk',
version='0.0.1',
author='Damir Cavar, Billy Dickson, Koushik Reddy Parukola',
author_email='dcavar@iu.edu, dicksonb@iu.edu, koparu@iu.edu',
description='A Quantum Natural Language Processing (Q-NLP) package',
packages=find_packages(),
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.6',
)
