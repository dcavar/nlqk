# coding: utf-8

"""
Natural Language Qu Kit (NLQK) - Quantum Natural Language Processing (QNLP) Library

(C) 2024-2025 by [Damir Cavar](http://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

"""


from sys import platform
from pathlib import Path
from nlqk import defaults


def get_data_folder() -> Path:
    """Get the path to the data folder for NLQK.
    As a side-effect the folder is created if it does not exist.
    Returns:
            """
    if platform == "linux" or platform == "linux2":
        data_directory = Path.home() / defaults.DATA_FOLDER_NAME
    elif platform == "darwin":
        data_directory = Path.home() / defaults.DATA_FOLDER_NAME
    elif platform == "win32":
        data_directory = Path.home() / defaults.DATA_FOLDER_WIN / defaults.DATA_FOLDER_NAME
    else:
        data_directory = Path(".")

    # check whether folder exists
    if not data_directory.exists():
        data_directory.mkdir(parents=True)

    return data_directory


