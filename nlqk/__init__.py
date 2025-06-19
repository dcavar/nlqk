# coding: utf-8

"""
Natural Language Qu Kit (NLQK) - Quantum Natural Language Processing (QNLP) Library

(C) 2024-2025 by [Damir Cavar](http://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

"""


from sys import platform
from pathlib import Path
import os


DATA_FOLDER_NAME = "nlqk_data"



def get_corpus_folder() -> Path:

    if platform == "linux" or platform == "linux2":
        # home/nltk_data
        # linux
        data_directory = Path.home() / DATA_FOLDER_NAME
    elif platform == "darwin":
        # OS X
        # like linux
        data_directory = Path.home() / DATA_FOLDER_NAME
    elif platform == "win32":
        # Windows...
        data_directory = Path.home() / "AppData" / "Roaming" / DATA_FOLDER_NAME
    else:
        data_directory = "."

    # check whether folder exists
    if not data_directory.exists():
        data_directory.mkdir(parents=True)

    return data_directory


