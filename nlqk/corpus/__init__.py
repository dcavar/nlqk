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


def download_simlex999() -> bool:
    # make sure there is a "corpora" subfolder in the data folder
    data_folder = nlqk.get_data_folder() / "corpora"
    simlex_folder = data_folder / "SimLex-999"
    if not data_folder.exists():
        data_folder.mkdir(parents=True)
    simlex_file = data_folder / "SimLex-999.zip"
    if simlex_file.exists():
        if simlex_folder.exists():
            return True
        # unzip zip file
        with zipfile.ZipFile(simlex_file, mode='r') as zip_ref:
            zip_ref.extractall(simlex_folder)
        return True
    # if the zip and folder do not exist
    data_url = "https://fh295.github.io/SimLex-999.zip"
    # unzip into the folder with the same name
    try:
        response = requests.get(data_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        with open(simlex_file, mode='wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(simlex_file, mode='r') as zip_ref:
            zip_ref.extractall(simlex_folder)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

