"""Miscellaneous methods."""

from contextlib import suppress
import os.path
import re

import firebase_admin
from firebase_admin import credentials, firestore
from gensim.models import KeyedVectors
from google.api_core import exceptions
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
        w2v = KeyedVectors.load_word2vec_format(os.path.join(), binary=True)
        w2v.init_sims(replace=True)
        w2v.save(path)
    return w2v


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
            try:
                yield tokenize(tweet), labels[int(line[5])]
            except:  # catch the last line
                ...


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


def timeout_stream(collection_ref, sleep_time=60, verbose=True):
    """Add a timeout handler for firebase collection streams."""
    stream = iter(cref.stream())
    with suppress(StopIteration):
        while True:
            try:
                yield next(stream)
            except exceptions.DeadlineExceeded:
                if verbose:
                    print(f"Caught `DeadlineExceeded` error, sleeping {sleep_time}s...")
                time.sleep(sleep_time)


def timeout_do(method, doc_ref, sleep_time=60, verbose=True, *args):
    """Add a timeout handler for an action on an individual firebase document."""
    try:
        method = getattr(doc_ref, method)
        if not len(args):
            return method()
        return method(*args)
    except exceptions.DeadlineExceeded:
        if verbose:
            print(f"Caught `DeadlineExceeded` error, sleeping {sleep_time}s...")
        time.sleep(sleep_time)
        timeout_write(doc_ref, value)
