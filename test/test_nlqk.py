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


DATA_FOLDER_NAME = nlqk.DATA_FOLDER_NAME # "nlqk_data"



class NLQKTest(unittest.TestCase):
    """
    """

    def test_get_corpus_folder(self):
        """
        """
        if sys.platform == "linux" or sys.platform == "linux2":
            data_directory = Path.home() / DATA_FOLDER_NAME
        elif sys.platform == "darwin":
            data_directory = Path.home() / DATA_FOLDER_NAME
        elif sys.platform == "win32":
            data_directory = Path.home() / "AppData" / "Roaming" / DATA_FOLDER_NAME
        else:
            data_directory = "."
        self.assertEqual(data_directory, nlqk.get_corpus_folder())


    def test_square_not_a_number(self):
        """
        If your put anything but a number,
        a TypeError Exception will be launch and your program will
        stop working.
        Usually you handle exception by better controlling inputs or
        using try and catch.
        """
        #with self.assertRaises(TypeError):
        #    square("not a number")
        pass


if __name__ == "__main__":
    unittest.main()
