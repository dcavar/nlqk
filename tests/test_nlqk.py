#!/usr/bin/env python3

"""
test_nlqk_corpus.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK corpus functionality.

"""


import sys
sys.path.append('..') # path to the module folder
import os
import unittest
from pathlib import Path
import numpy as np


class NLQKTestLocal(unittest.TestCase):
    """Testing the NLQK local library functionality."""

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "This test is deactivated on GitHub Actions")
    def test_get_corpus_folder(self):
        """Testing the get_data_folder function to return the correct data directory based on the platform."""
        from nlqk.defaults import DATA_FOLDER_NAME, DATA_FOLDER_WIN
        from nlqk import get_data_folder
        if sys.platform == "linux" or sys.platform == "linux2":
            data_directory = Path.home() / DATA_FOLDER_NAME
        elif sys.platform == "darwin":
            data_directory = Path.home() / DATA_FOLDER_NAME
        elif sys.platform == "win32":
            data_directory = Path.home() / DATA_FOLDER_WIN / DATA_FOLDER_NAME
        else:
            data_directory = Path(".")
        self.assertEqual(data_directory, get_data_folder())


class NLQKTestEmbeddings(unittest.TestCase):
    """Testing the embeddings module."""

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "This test is deactivated on GitHub Actions")
    def test_is_normalized(self):
        """Test the is_normalized function."""
        from nlqk.embedding import is_normalized
        v = np.random.rand(8)
        v_norm = v / np.linalg.norm(v)
        #print(embedding.is_normalized(v_norm))
        self.assertTrue(is_normalized(v_norm))

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "This test is deactivated on GitHub Actions")
    def test_normalize(self):
        """Test the normalize function."""
        from nlqk.embedding import normalize
        v = np.random.rand(8)
        v_norm = v / np.linalg.norm(v)
        self.assertTrue(np.array_equal(v_norm, normalize(v)))


if __name__ == "__main__":
    unittest.main()
