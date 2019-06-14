import os

import numpy as np
from scipy.spatial.distance import cosine as cos_dist
import tensorflow as tf

import utils


def norm_dist(u, v):
    return np.linalg.norm(u / np.linalg.norm(u) - v / np.linalg.norm(v))


if __name__ == "__main__":
    w2v = utils.load_word2vec()
    avg_embed = lambda vs: sum(vs) / len(vs)

    if not os.path.exists("projector"):
        os.makedirs("projector")

    embedding_matrix = []
    stormfront_vals = [[], []]
    twitter_vals = [[], []]
    with open(os.path.join("projector", "metadata.tsv"), "w") as f:
        f.write("label\tcontent\n")

        for tokens, label in utils.stormfront_gen():
            acc = []
            for token in tokens:
                if token in w2v:
                    stormfront_vals[label].append(w2v[token])
                    acc.append(w2v[token])
            if len(acc) > 0:
                f.write(f"{label}\t{' '.join(tokens)}\n")
                embedding_matrix.append(avg_embed(acc))
        stormfront_vals = list(map(avg_embed, stormfront_vals))

        for tokens, label in utils.twitter_gen():
            acc = []
            for token in tokens:
                if token in w2v:
                    twitter_vals[label].append(w2v[token])
                    acc.append(w2v[token])
            if len(acc) > 0:
                f.write(f"{label}\t{' '.join(tokens)}\n")
                embedding_matrix.append(avg_embed(acc))
        twitter_vals = list(map(avg_embed, twitter_vals))

        youtube_vals = []
        rf = open(os.path.join("data", "youtube_samples.txt"))
        for line in rf:
            acc = []
            tokens = utils.tokenize(line)
            for token in tokens:
                if token in w2v:
                    youtube_vals.append(w2v[token])
                    acc.append(w2v[token])
            if len(acc) > 0:
                f.write(f"2\t{' '.join(tokens)}\n")
                embedding_matrix.append(avg_embed(acc))
        youtube_vals = avg_embed(youtube_vals)
        rf.close()

    g_to_np = lambda x: np.array(list(x))
    sn = g_to_np(stormfront_vals[0])
    sh = g_to_np(stormfront_vals[1])
    tn = g_to_np(twitter_vals[0])
    th = g_to_np(twitter_vals[1])
    y = g_to_np(youtube_vals)

    print("Distances between normalized vectors")
    print("||sn - sh|| =", norm_dist(sn, sh))
    print("||tn - th|| =", norm_dist(tn, th))
    print("||sn - tn|| =", norm_dist(sn, tn))
    print("||sh - th|| =", norm_dist(sh, th))
    print("||(sn + tn) - (sh + th)|| =", norm_dist(sn + tn, sh + th))
    print("||(sn + sh) - (tn + th)|| =", norm_dist(sn + sh, tn + th))
    print("||y - sn|| = ", norm_dist(y, sn))
    print("||y - sh|| = ", norm_dist(y, sh))
    print("||y - tn|| = ", norm_dist(y, tn))
    print("||y - th|| = ", norm_dist(y, th))
    print("||y - (sn + sh)|| = ", norm_dist(y, sn + sh))
    print("||y - (tn + th)|| = ", norm_dist(y, tn + th))
    print("||y - (sn + tn)|| = ", norm_dist(y, sn + tn))
    print("||y - (sh + th)|| = ", norm_dist(y, sh + th))
    print()

    print("Cosine distances")
    print("cos_dist(sn, sh) = ", cos_dist(sn, sh))
    print("cos_dist(tn, th) = ", cos_dist(tn, th))
    print("cos_dist(sn, tn) = ", cos_dist(sn, tn))
    print("cos_dist(sh, th) = ", cos_dist(sh, th))
    print("cos_dist(sn + tn, sh + th) = ", cos_dist(sn + tn, sh + th))
    print("cos_dist(sn + sh, tn + th) = ", cos_dist(sn + sh, tn + th))
    print("cos_dist(y, sn) = ", cos_dist(y, sn))
    print("cos_dist(y, sh) = ", cos_dist(y, sh))
    print("cos_dist(y, tn) = ", cos_dist(y, tn))
    print("cos_dist(y, th) = ", cos_dist(y, th))
    print("cos_dist(y, sn + sh) = ", cos_dist(y, sn + sh))
    print("cos_dist(y, tn + th) = ", cos_dist(y, tn + th))
    print("cos_dist(y, sn + tn) = ", cos_dist(y, sn + tn))
    print("cos_dist(y, sh + th) = ", cos_dist(y, sh + th))
    print()

    with open(os.path.join("projector", "embeddings.tsv"), "w") as f:
        for embed in embedding_matrix:
            f.write("\t".join(list(map(str, embed))) + "\n")

    print(
        "Ready! Go to http://projector.tensorflow.org/ and upload the files in the "
        "`projector` directory to visualize"
    )
