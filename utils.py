"""Miscellaneous methods."""

import os.path
import re

from gensim.models import KeyedVectors


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
