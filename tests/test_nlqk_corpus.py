#!/usr/bin/env python3

"""
test_nlqk_corpus.py

(C) 2025 by Damir Cavar

Testing the NLQK corpus functionality.
"""


import unittest
from pathlib import Path
import sys
sys.path.append('..') # path to the module folder

import nlqk



class NLQKCorpusTest(unittest.TestCase):
    """
    """

    def test_get_SimLex(self):
        """
        """
        #if sys.platform == "linux" or sys.platform == "linux2":
        #    data_directory = Path.home() / DATA_FOLDER_NAME
        #elif sys.platform == "darwin":
        #    data_directory = Path.home() / DATA_FOLDER_NAME
        #elif sys.platform == "win32":
        #    data_directory = Path.home() / "AppData" / "Roaming" / DATA_FOLDER_NAME
        #else:
        #    data_directory = "."
        #self.assertEqual(data_directory, nlqk.get_corpus_folder())
        pass


if __name__ == "__main__":
    unittest.main()

