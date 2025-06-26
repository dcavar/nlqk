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
import pandas as pd
import nlqk.defaults # nlqk
# import nlqk #. # as nlqk


def download_simlex999() -> bool:
    """Download the SimLex-999 dataset and extract it to the appropriate folder.
    Returns:
        bool: True if the download and extraction were successful, False otherwise.
    """
    # make sure there is a corpora subfolder in the data folder
    data_folder = nlqk.get_data_folder() / nlqk.defaults.DATA_FOLDER_CORPORA
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
        # return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    # unpack the nouns
    simlex_main_file = data_folder / nlqk.defaults.SIMLEX_999_FOLDER / nlqk.defaults.SIMLEX_999_FILE
    try:
        if simlex_main_file.exists():
            with open(simlex_main_file, mode='r', encoding='utf-8') as f:
                # Process the file if needed
                df = pd.read_csv(f, sep='\t', header=0, encoding='utf-8')
                df_noun_pairs = df.loc[df['POS'] == 'N']
                df_noun_pairs.to_csv(data_folder / nlqk.defaults.SIMLEX_999_FOLDER / 'nouns_data.txt', sep='\t', index=False, encoding='utf-8')
                string_values_list = list(set(df_noun_pairs['word1'].astype(str).tolist() + df_noun_pairs['word2'].astype(str).tolist()))
                string_values_list.sort()
                with open(data_folder / nlqk.defaults.SIMLEX_999_FOLDER / 'nouns.txt', mode='w', encoding='utf-8') as nouns_file:
                    for value in string_values_list:
                        nouns_file.write(value + '\n')
                return True
        else:
            print(f"SimLex-999 file not found: {simlex_main_file}")
            return False
    except IOError as e:
        print(f"Error accessing file: {e}")
        return False
    return True

