# coding: utf-8

"""
Natural Language Qu Kit (NLQK) - Quantum Natural Language Processing (QNLP) Library

Corpus package

(C) 2025 by [Damir Cavar](http://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

"""


import requests
from sys import platform
from pathlib import Path
import zipfile
import nlqk
import nlqk.defaults


def download_simlex999() -> bool:
    """Download the SimLex-999 dataset and extract it to the appropriate folder.
    Returns:
        bool: True if the download and extraction were successful, False otherwise.
    """
    # make sure there is a "corpora" subfolder in the data folder
    data_folder = nlqk.get_data_folder() / "corpora"
    if not data_folder.exists():
        data_folder.mkdir(parents=True)
    simlex_file = data_folder / nlqk.defaults.SIMLEX_999_ZIP_FILE
    if simlex_file.exists():
        if data_folder.exists():
            return True
        with zipfile.ZipFile(simlex_file, mode='r') as zip_ref:
            zip_ref.extractall(data_folder)
        return True
    try:
        response = requests.get(nlqk.defaults.SIMLEX_999_URL, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        with open(simlex_file, mode='wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(simlex_file, mode='r') as zip_ref:
            zip_ref.extractall(data_folder)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False



