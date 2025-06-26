#!/usr/bin/env python3

"""
test_nlqk_corpus.py

(C) 2025 by Damir Cavar

Testing the NLQK corpus functionality.
"""


import sys
sys.path.append('..') # path to the module folder
import os
import unittest
from pathlib import Path


class NLQKCorpusTest(unittest.TestCase):
    """Testing the NLQK corpus functionality."""

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true", "This test is deactivated on GitHub Actions")
    def test_get_SimLex(self):
        """Testing the SimLex corpus download functionality."""
        from nlqk.corpus import download_simlex999
        self.assertEqual(True, download_simlex999())




if __name__ == "__main__":
    unittest.main()

