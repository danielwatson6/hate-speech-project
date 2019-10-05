import json
import os
import sys

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf


MF_KEYS = ["authority", "fairness", "care", "loyalty", "purity", "non_moral"]


mf = None
ambiguity = None


def print_correlations(pairs):
    mf_vecs, ambg_scores = zip(*pairs)
    mf_vecs = np.array(mf_vecs).T
    for mf_dim_name, mf_dim_vals in zip(MF_KEYS, mf_vecs):
        print(f"{mf_dim_name}/ambiguity")
        print("  pcc {:.4f} (p={:.4f})".format(*pearsonr(mf_dim_vals, ambg_scores)))
        print("  scc {:.4f} (p={:.4f})".format(*spearmanr(mf_dim_vals, ambg_scores)))


def wordnet_score(x):
    scores = []
    for word in word_tokenize:
        scores.append(len(wn.synsets(x)))
    return sum(scores) / len(scores)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.wordnet_mf [mf_save_dir]", file=sys.stderr)
        exit(1)

    # Instantiate and read weights of a trained moral foundations model.
    save_dir = sys.argv[1]
    runpy_json_path = os.path.join("experiments", save_dir, "runpy.json")
    with open(runpy_json_path) as f:
        module = json.load(f)["model"]

    MFModel = getattr(__import__(f"models.{module}", {module}))
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

    # ============================================================= #
    # Correlation for twitter moral foundations validation dataset. #
    # ============================================================= #

    twitter_mf = tf.data.experimental.make_csv_dataset(
        os.path.join("data", "twitter_mf.clean.shuffled.csv"),
        mf.hparams.batch_size,
        num_epochs=1,
        shuffle=False,
        num_rows_for_inference=None,
    )

    def _mf_gold_map(batch):
        tweets = tf.strings.split(batch["tweet"]).to_tensor(default_value="<pad>")
        labels = [batch[k] for k in MF_KEYS]
        labels = tf.stack(labels)
        labels = tf.cast(labels, tf.float32)
        labels = tf.transpose(labels)
        return tweets, labels

    twitter_mf_gold_pairs = []
    twitter_mf_gold = twitter_mf.map(_mf_gold_map)
    for xs, ys in twitter_mf_gold:
        for x, y in zip(xs, ys):
            twitter_mf_gold_pairs.append(y, wordnet_score(x))

    print("Twitter MF gold vs. wordnet:")
    print_correlations(twitter_mf_gold_pairs)

    def _mf_valid_map(batch):
        tweets = tf.strings.split(batch["tweet"]).to_tensor(default_value="<pad>")
        return tweets, word_to_id(tweets)

    twitter_mf_valid_pairs = []
    twitter_mf_valid = twitter_mf.take(mf.hparams.num_valid // mf.hparams.batch_size)
    twitter_mf_valid = twitter_mf_valid.map(_mf_valid_map)
    for xs, xs_ids in twitter_mf_valid:
        ys = mf(xs_ids)
        for x, y in zip(xs, ys):
            twitter_mf_valid_pairs.append(y, wordnet_score(x))

    print("\nTwitter MF validation output vs. wordnet:")
    print_correlations(twitter_mf_valid_pairs)
