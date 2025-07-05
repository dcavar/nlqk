#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
embedding.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/) and the [NLP Lab](https://nlp-lab.org/)

Module: nlqk.embeddings.states


"""

import os
from typing import List
try: # prefer RAPIDS libraries and GPU over numpy and CPU
    import cupy as np  # Try to import cupy and alias it as np
    _USE_GPU = True
except ModuleNotFoundError:
    import numpy as np  # If cupy not found, import numpy and alias it as np
    _USE_GPU = False
from nlqk.defaults import OPEN_AI_EMBEDDING_MODELS


#from .vectors import (
#    is_normalized,
#    normalize,
#    pad_vector,
#    pair_real_to_complex,
#    cosine_similarity,
#)
#from .states import (
#    hamiltonian_to_state,
#    check_states_equal,
#)


def get_openai_embeddings(wordlist: List[str], api_key = '', model_name: str = 'large') -> np.ndarray:
    """Get the GPT embeddings for a wordlist.

    Args:
        wordlist List of str: List of words.
        api_key str: The OpenAI API key.
        model_name str: One of the valid OpenAI embedding model names.

    Returns:
        np.ndarray: the OpenAI embeddings for the words in the wordlist.

    Raises:
        ValueError: If the OpenAI key is missing, i.e., no specification of arg 'api_key' and not environment variable OPENAI_API_KEY.
    """

    if model_name not in OPEN_AI_EMBEDDING_MODELS:
        raise ValueError(f'model_name not a valid OpenAI embedding model name. Use one of: {", ".join(OPEN_AI_EMBEDDING_MODELS.keys())}')
    if not api_key:
        # check environment variable OPENAI_API_KEY
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(f"Attempted OpenAI API call without API-key. Provide a valid value for 'api_key' or set the API-key in the environment variable OPENAI_API_KEY.")
    # TODO call openai api
    return np.array([])


def get_embeddings(wordlist: List[str], ) -> np.ndarray:

    return np.array([])

