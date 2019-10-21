import json
import os
import re
import sys

import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import numpy as np
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf


MF_KEYS = ["authority", "fairness", "care", "loyalty", "purity", "non_moral"]


def plot_correlations(pairs, title):
    mf_vecs, ambg_scores = zip(*pairs)
    mf_vecs = np.array(mf_vecs).T

    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.5)

    for i, (mf_dim_name, mf_dim_vals) in enumerate(zip(MF_KEYS, mf_vecs), 1):
        pcc = pearsonr(mf_dim_vals, ambg_scores)
        scc = spearmanr(mf_dim_vals, ambg_scores)

        plt.subplot(3, 2, i)
        plt.plot(ambg_scores, mf_dim_vals, "o", color="black", markersize=2)
        plt.xlabel(
            "ambiguity\npcc={:.4f} (p={:.4f}) scc={:.4f} (p={:.4f})".format(
                *(pcc + scc)
            )
        )
        plt.ylabel(mf_dim_name)
        plt.xlim(0 - 1.0, 50 + 1.0)
        plt.ylim(-1 - 0.1, 1 + 0.1)

    plt.show()


def wordnet_score(x, reduction="mean"):
    scores = []
    for word in x.numpy():
        word = word.decode("utf8")
        if word == "ACCOUNT":
            continue
        if not re.match(r"[A-Za-z'-]+", word):
            continue
        n = len(wn.synsets(word))
        if n > 0:
            scores.append(n)

    if len(scores) == 0:
        return 0

    if reduction == "median":
        return np.median(scores)
    mean = sum(scores) / len(scores)
    if reduction == "mean":
        return mean
    return sum(abs(x - mean) for x in scores) / len(scores)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m scripts.wordnet_mf [mf_save_dir] [video_segment_len]?",
            file=sys.stderr,
        )
        exit(1)

    # Instantiate and read weights of a trained moral foundations model.
    save_dir = sys.argv[1]
    runpy_json_path = os.path.join("experiments", save_dir, "runpy.json")
    with open(runpy_json_path) as f:
        module = json.load(f)["model"]

    MFModel = getattr(__import__(f"models.{module}"), module)
    mf = MFModel(os.path.join("experiments", save_dir))
    mf.restore()

    # Any dataset fed into the model needs to use the original word/id mapping.
    vocab_path = os.path.join("data", "twitter_mf.clean.vocab")
    word_to_id_init = tf.lookup.TextFileInitializer(
        vocab_path,
        tf.string,
        tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER,
        vocab_size=mf.hparams.vocab_size,
    )
    word_to_id = tf.lookup.StaticHashTable(word_to_id_init, 1).lookup

    # =================================================== #
    # Correlations for twitter moral foundations dataset. #
    # =================================================== #

    # twitter_mf = tf.data.experimental.make_csv_dataset(
    #     os.path.join("data", "twitter_mf.clean.shuffled.csv"),
    #     mf.hparams.batch_size,
    #     num_epochs=1,
    #     shuffle=False,
    #     num_rows_for_inference=None,
    # )

    # # We include correlations for the moral labels of the entire data.

    # def _mf_gold_map(batch):
    #     tweets = tf.strings.split(batch["tweet"]).to_tensor(default_value="<pad>")
    #     labels = [batch[k] for k in MF_KEYS]
    #     labels = tf.stack(labels)
    #     labels = tf.cast(labels, tf.float32)
    #     labels = tf.transpose(labels)
    #     return tweets, labels

    # twitter_mf_gold_pairs = []
    # twitter_mf_gold = twitter_mf.map(_mf_gold_map)
    # for xs, ys in twitter_mf_gold:
    #     for x, y in zip(xs, ys):
    #         twitter_mf_gold_pairs.append([y, wordnet_score(x)])

    # plot_correlations(twitter_mf_gold_pairs, "Twitter MF gold vs. wordnet")

    # # But also correlations for the inferred model's scores on the validation data.

    # def _mf_valid_map(batch):
    #     tokens = tf.strings.split(batch["tweet"]).to_tensor(default_value="<pad>")
    #     return tokens, word_to_id(tokens)

    # twitter_mf_valid_pairs = []
    # twitter_mf_valid = twitter_mf.take(mf.hparams.num_valid // mf.hparams.batch_size)
    # twitter_mf_valid = twitter_mf_valid.map(_mf_valid_map)
    # for xs, xs_ids in twitter_mf_valid:
    #     ys = mf(xs_ids)
    #     for x, y in zip(xs, ys):
    #         twitter_mf_valid_pairs.append([y, wordnet_score(x)])

    # plot_correlations(
    #     twitter_mf_valid_pairs, "Twitter MF validation output vs. wordnet"
    # )

    # =================================================== #
    # Correlations for YouTube comment and video samples. #
    # =================================================== #

    def _youtube_map(batch):
        tokens = tf.strings.split(batch).to_tensor(default_value="<pad>")
        return tokens, word_to_id(tokens)

    # comments = tf.data.TextLineDataset(
    #     os.path.join("data", "youtube_comment_samples.txt")
    # )
    # comments = comments.batch(mf.hparams.batch_size).map(_youtube_map)

    # youtube_comment_pairs = []
    # for xs, xs_ids in comments:
    #     ys = mf(xs_ids)
    #     for x, y in zip(xs, ys):
    #         youtube_comment_pairs.append([y, wordnet_score(x)])

    # plot_correlations(youtube_comment_pairs, "Youtube comment samples vs. wordnet")

    videos = tf.data.TextLineDataset(os.path.join("data", "youtube_video_samples.txt"))

    # Optional segmentation.
    if len(sys.argv) == 3:
        segment_len = int(sys.argv[2])

        def segment_video(v):
            vlen = tf.strings.length(v)
            pos = tf.range(vlen, delta=segment_len)
            lens = tf.tile([segment_len], tf.shape(pos))
            return tf.data.Dataset.from_tensor_slices(tf.strings.substr(v, pos, lens))

        videos = videos.flat_map(segment_video)

    videos = videos.batch(mf.hparams.batch_size).map(_youtube_map)

    youtube_video_pairs = []
    for xs, xs_ids in videos:
        ys = mf(xs_ids)
        for x, y in zip(xs, ys):
            youtube_video_pairs.append([y, wordnet_score(x)])

    plot_correlations(youtube_video_pairs, "Youtube video samples vs. wordnet")
