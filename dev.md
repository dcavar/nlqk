# NLQK - Natural Language Qu Kit - Python Quantum NLP Tools

(C) 2024 by [Damir Cavar], NLP-Lab


## Prerequisites

Install twine and tox

    pip install -U twine
    pip install -U tox


## Build the Package

Run the following command:

    python setup.py sdist


## Upload Package to Pypi

Test whether the package can be build correctly:

     pip install -e .

Run the following command:

    twine upload dist/*

Use `tox` to validate the code.

