import os

import numpy as np
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf

import utils


w2v = {}
embedding_matrix = []


def embeds_and_meta(gen, label0, label1, meta_file):
    avg_embed = lambda vs: sum(vs) / len(vs)
    vals = [[], []]
    for tokens, label in gen():
        acc = []
        for token in tokens:
            if token in w2v:
                vals[label].append(w2v[token])
                acc.append(w2v[token])

        if len(acc) > 0:
            label = [label0, label1][label]
            text = " ".join(tokens)
            if len(text) >= 125:
                text = text[:125] + "..."
            meta_file.write(f"{label}\t{text}\n")
            embedding_matrix.append(avg_embed(acc))

    return list(map(avg_embed, vals))


def norm_dist(u, v):
    return np.linalg.norm(u / np.linalg.norm(u) - v / np.linalg.norm(v))


if __name__ == "__main__":
    w2v = utils.load_word2vec()
    avg_embed = lambda vs: sum(vs) / len(vs)

    if not os.path.exists("projector"):
        os.makedirs("projector")

    stormfront_vals = [[], []]
    twitter_vals = [[], []]
    with open(os.path.join("projector", "metadata.tsv"), "w") as f:
        f.write("label\tcontent\n")

        # for tokens, label in utils.stormfront_gen():
        #     acc = []
        #     for token in tokens:
        #         if token in w2v:
        #             stormfront_vals[label].append(w2v[token])
        #             acc.append(w2v[token])
        #     if len(acc) > 0:
        #         label = ["sn", "sh"][label]
        #         f.write(f"{label}\t{' '.join(tokens)}\n")
        #         embedding_matrix.append(avg_embed(acc))
        # stormfront_vals = list(map(avg_embed, stormfront_vals))

        # for tokens, label in utils.twitter_gen():
        #     acc = []
        #     for token in tokens:
        #         if token in w2v:
        #             twitter_vals[label].append(w2v[token])
        #             acc.append(w2v[token])
        #     if len(acc) > 0:
        #         label = ["tn", "th"][label]
        #         f.write(f"{label}\t{' '.join(tokens)}\n")
        #         embedding_matrix.append(avg_embed(acc))
        # twitter_vals = list(map(avg_embed, twitter_vals))

        stormfront_vals = embeds_and_meta(utils.stormfront_gen, "sn", "sh", f)
        twitter_vals = embeds_and_meta(utils.twitter_gen, "tn", "th", f)
        youtube_vals = embeds_and_meta(utils.youtube_samples_gen, "yv", "yc", f)

    g_to_np = lambda x: np.array(list(x))
    sn = g_to_np(stormfront_vals[0])
    sh = g_to_np(stormfront_vals[1])
    tn = g_to_np(twitter_vals[0])
    th = g_to_np(twitter_vals[1])
    yv = g_to_np(youtube_vals[0])
    yc = g_to_np(youtube_vals[1])

    pairs = [
        ("sn, sh", sn, sh),
        ("tn, th", tn, th),
        ("sn, tn", sn, tn),
        ("sh, th", sh, th),
        ("sn + tn, sh + th", sn + tn, sh + th),
        ("sn + sh, tn + th", sn + sh, tn + th),
        ("yv, yc", yv, yc),
        ("yv, sn", yv, sn),
        ("yv, sh", yv, sh),
        ("yv, tn", yv, tn),
        ("yv, th", yv, th),
        ("yv, sn + sh", yv, sn + sh),
        ("yv, tn + th", yv, tn + th),
        ("yv, sn + tn", yv, sn + tn),
        ("yv, sh + th", yv, sh + th),
        ("yc, sn", yc, sn),
        ("yc, sh", yc, sh),
        ("yc, tn", yc, tn),
        ("yc, th", yc, th),
        ("yc, sn + sh", yc, sn + sh),
        ("yc, tn + th", yc, tn + th),
        ("yc, sn + tn", yc, sn + tn),
        ("yc, sh + th", yc, sh + th),
        ("yv + yc, sn", yv + yc, sn),
        ("yv + yc, sh", yv + yc, sh),
        ("yv + yc, tn", yv + yc, tn),
        ("yv + yc, th", yv + yc, th),
        ("yv + yc, sn + sh", yv + yc, sn + sh),
        ("yv + yc, tn + th", yv + yc, tn + th),
        ("yv + yc, sn + tn", yv + yc, sn + tn),
        ("yv + yc, sh + th", yv + yc, sh + th),
    ]

    scores = np.zeros((4, len(pairs)))
    for i, (s, x, y) in enumerate(pairs):
        scores[0][i] = norm_dist(x, y)
        scores[2][i] = cos_dist(x, y)

    scores[1] = scores[0] / max(scores[0])
    scores[3] = scores[2] / max(scores[2])

    print("Distances between normalized vectors")
    for count, i in enumerate(np.argsort(scores[3]), 1):
        print(f"{pairs[i][0]} ({count}/{len(pairs)})")
        print("norm\t{0:.4f}\t{1:.4f}".format(scores[0][i], scores[1][i]))
        print("cos\t{0:.4f}\t{1:.4f}\n".format(scores[2][i], scores[3][i]))

    print(
        "Ready! Go to http://projector.tensorflow.org/ and upload the files in the "
        "`projector` directory to visualize the embeddings.\n"
    )

    pearson = pearsonr(scores[1], scores[3])
    print("pearson correlation\t{0:.4f}\t(p={1:.4f})".format(pearson[0], pearson[1]))
    spearman = spearmanr(scores[1], scores[3])
    print("spearman correlation\t{0:.4f}\t(p={1:.4f})".format(spearman[0], spearman[1]))
