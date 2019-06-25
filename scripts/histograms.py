"""."""

from argparse import ArgumentParser
from collections import Counter
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def make_word_to_id(vocab_size):
    word_freqs = Counter()  # word -> # occurences in corpus
    for words, label in utils.chain(utils.stormfront_gen, utils.twitter_gen)():
        for word in words:
            word_freqs[word] += 1
    id_to_word_and_freq = word_freqs.most_common(min(vocab_size, len(word_freqs)))
    return {word: i for i, (word, _) in enumerate(id_to_word_and_freq)}


def histograms(gen, word_to_id, tf_idf=False):
    # No-hate and hate document frequencies.
    doc_counts = [0, 0]
    doc_freqs = [Counter(), Counter()]  # word -> # docs containing word

    for words, label in gen():
        doc_counts[label] += 1
        words_in_doc = {}
        for word in words:
            if word not in words_in_doc:
                words_in_doc[word] = None
                doc_freqs[label][word] += 1

    # No-hate and hate histograms.
    histos = np.zeros((2, len(word_to_id)))

    for words, label in gen():
        for word in words:
            # Remove OOV words.
            if word not in word_to_id:
                continue
            id_ = word_to_id[word]
            inc = 1
            if tf_idf:
                tf = words.count(word) / len(words)
                idf = np.log(doc_counts[label] / doc_freqs[label][word])
                inc = tf * idf
            histos[label][id_] += inc

    if not np.equal(0, histos[0])[0]:
        histos[0] /= sum(histos[0])
    if not np.equal(0, histos[1])[0]:
        histos[1] /= sum(histos[1])
    return histos


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--tf_idf", type=bool, default=False)
    parser.add_argument("--merge", type=bool, default=False)
    FLAGS = parser.parse_args()

    word_to_id = make_word_to_id(FLAGS.vocab_size)
    x_values = list(range(len(word_to_id)))

    if FLAGS.merge:
        histos = histograms(
            utils.chain(utils.stormfront_gen, utils.twitter_gen),
            word_to_id,
            tf_idf=FLAGS.tf_idf,
        )
        ymax = np.amax(histos)
        plt.subplot(2, 2, 1)
        plt.plot(x_values, histos[0], "g-")
        plt.ylim(0, ymax)
        plt.subplot(2, 2, 2)
        plt.plot(x_values, histos[1], "r-")
        plt.ylim(0, ymax)
        plt.show()
    else:
        stormfront_histos = histograms(
            utils.stormfront_gen, word_to_id, tf_idf=FLAGS.tf_idf
        )
        twitter_histos = histograms(utils.twitter_gen, word_to_id, tf_idf=FLAGS.tf_idf)
        ymax = max(np.amax(stormfront_histos), np.amax(twitter_histos))
        plt.subplot(2, 2, 1)
        plt.plot(x_values, stormfront_histos[0], "g-")
        plt.ylim(0, ymax)
        plt.subplot(2, 2, 2)
        plt.plot(x_values, stormfront_histos[1], "r-")
        plt.ylim(0, ymax)
        plt.subplot(2, 2, 3)
        plt.plot(x_values, twitter_histos[0], "g-")
        plt.ylim(0, ymax)
        plt.subplot(2, 2, 4)
        plt.plot(x_values, twitter_histos[1], "r-")
        plt.ylim(0, ymax)
        plt.show()
