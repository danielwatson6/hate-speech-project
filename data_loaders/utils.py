"""Miscellaneous functions used exclusively by data loaders."""

import os

import numpy as np
import tensorflow as tf

import utils


def save_or_load_embeds(embeds_path, vocab_path, vocab_size):
    """Load or build and embedding matrix from a TSV file."""

    should_save = False

    if not os.path.isfile(embeds_path):
        embedding_matrix = np.random.uniform(low=-1.0, high=1.0, size=(vocab_size, 300))
        should_save = True
    else:
        embedding_matrix = np.load(embeds_path)
        # Check if the vocab sizes match. If the saved matrix is missing words, a new
        # matrix file is needed.
        if len(embedding_matrix) < vocab_size:
            should_save = True

    if should_save:
        w2v = utils.load_word2vec()
        with open(vocab_path) as f:
            for i, word in enumerate(f):
                word = word.strip()
                if word in w2v:
                    embedding_matrix[i] = w2v[word]

        np.save(embeds_path, embedding_matrix)

    return embedding_matrix
