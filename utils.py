"""Miscellaneous methods."""

from contextlib import suppress
import os.path
import re
import time

import firebase_admin
from firebase_admin import credentials, firestore
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd


def tokenize(s):
    """Elementary string cleaning and tokenizing."""
    s = re.sub(r"[^a-z]", " ", s.lower())  # lowercase and remove non-words
    s = re.sub(r"[\s]+", " ", s).strip()  # get rid of extra spacing
    return s.split()


def chain(*gen_fns):
    """Similar to `itertools.chain` but takes and returns callables."""

    def _chain():
        for it in gen_fns:
            for el in it():
                yield el

    return _chain


def load_word2vec():
    """Get a word2vec model, ensuring fast loading."""
    path = os.path.join("data", "word2vec")
    if os.path.exists(path):
        w2v = KeyedVectors.load(path, mmap="r")
    else:
        # TODO: fix this line.
        w2v = KeyedVectors.load_word2vec_format(os.path.join(), binary=True)
        w2v.init_sims(replace=True)
        w2v.save(path)
    return w2v


def save_or_load_embeds(embeds_path, vocab_path, vocab_size):
    if not os.path.isfile(embeds_path):
        word2vec = load_word2vec()
        embedding_matrix = np.random.uniform(low=-1.0, high=1.0, size=(vocab_size, 300))

        with open(vocab_path) as f:
            for i, word in enumerate(f):
                word = word.strip()
                if word in word2vec:
                    embedding_matrix[i] = word2vec[word]

        np.save(embeds_path, embedding_matrix)
        return embedding_matrix

    return np.load(embeds_path)


def stormfront_gen():
    """Generator for the stormfront dataset."""
    labels = {"noHate": 0, "hate": 1}
    with open(os.path.join("data", "stormfront.csv")) as f:
        f.readline()  # skip the csv header
        for line in f:
            line = line.split(",")  # line[0] = filename, line[4] = label
            label = line[4].strip()
            if label not in labels:
                continue
            with open(os.path.join("data", "stormfront", line[0] + ".txt")) as lf:
                phrase = lf.readline()
            yield tokenize(phrase), labels[label]


def twitter_gen():
    """Generator for the twitter dataset."""
    labels = [1, 0, 0]
    with open(os.path.join("data", "twitter.csv")) as f:
        f.readline()  # skip the csv header
        for line in f:
            line = line.split(",")
            tweet = ",".join(line[6:])
            with suppress(Exception):
                yield tokenize(tweet), labels[int(line[5])]


def youtube_samples_gen():
    """Generator for the youtube samples dataset."""
    with open(os.path.join("data", "youtube_video_samples.txt")) as f:
        for line in f:
            yield tokenize(line), 0
    with open(os.path.join("data", "youtube_comment_samples.txt")) as f:
        for line in f:
            yield tokenize(line), 1


def firebase(backup=False, verbose=True):
    """Return a firebase SDK client instance."""
    if backup:
        cred = credentials.Certificate(
            os.path.join("secrets", "online-extremism-backup.json")
        )
        app = firebase_admin.initialize_app(cred, name="online-extremism-backup")
    else:
        cred = credentials.Certificate(os.path.join("secrets", "online-extremism.json"))
        app = firebase_admin.initialize_app(cred)
    client = firestore.client(app)
    if verbose:
        print(f"Connected to Firebase SDK ({client.project})")
    return client


def timeout_stream(collection_ref, sleep_time=1800, verbose=True):
    """Add a timeout handler for firebase collection streams."""
    stream = iter(cref.stream())
    with suppress(StopIteration):
        while True:
            try:
                yield next(stream)
            except:
                if verbose:
                    print(f"Quota/deadline exceeded, sleeping {sleep_time}s...")
                time.sleep(sleep_time)


def timeout_do(method, doc_ref, args=None, sleep_time=1800, verbose=True):
    """Add a timeout handler for an action on an individual firebase document."""
    try:
        method = getattr(doc_ref, method)
        if args is None:
            return method()
        return method(*args)
    except:
        if verbose:
            print(f"Quota/deadline exceeded, sleeping {sleep_time}s...")
        time.sleep(sleep_time)
        timeout_do(method, doc_ref, args=args, sleep_time=sleep_time, verbose=verbose)
