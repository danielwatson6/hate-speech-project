import os

import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf

import utils


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
                f.write(f"{label}]\t{' '.join(tokens)}\n")
                embedding_matrix.append(avg_embed(acc))
        stormfront_vals = list(map(avg_embed, stormfront_vals))

        for tokens, label in utils.twitter_gen():
            acc = []
            for token in tokens:
                if token in w2v:
                    twitter_vals[label].append(w2v[token])
            if len(acc) > 0:
                f.write(f"{label}\t{' '.join(tokens)}\n")
                embedding_matrix.append(avg_embed(acc))
        twitter_vals = list(map(avg_embed, twitter_vals))

    g_to_np = lambda x: np.array(list(x))
    sn = g_to_np(stormfront_vals[0])
    sh = g_to_np(stormfront_vals[1])
    tn = g_to_np(twitter_vals[0])
    th = g_to_np(twitter_vals[1])

    print("Distances")
    print("||sn - sh|| =", np.linalg.norm(sn - sh))
    print("||tn - th|| =", np.linalg.norm(tn - th))
    print("||sn - tn|| =", np.linalg.norm(sn - tn))
    print("||sh - th|| =", np.linalg.norm(sh - th))
    print("||sn + tn - sh - th|| =", np.linalg.norm(sn + tn - sh - th))
    print("||sn + sh - sh - th|| =", np.linalg.norm(sn + tn - sh - th))
    print()

    print("Cosine distances")
    print("cos_dist(sn, sh)", cosine(sn, sh))
    print("cos_dist(tn, th)", cosine(tn, th))
    print("cos_dist(sn, tn)", cosine(sn, tn))
    print("cos_dist(sh, th)", cosine(sh, th))
    print("cos_dist(sn + tn, sh + th)", cosine(sn + tn, sh + th))
    print("cos_dist(sn + sh, tn + th)", cosine(sn + sh, tn + th))
    print()

    with open(os.path.join("projector", "embeddings.tsv"), "w") as f:
        for embed in embedding_matrix:
            f.write("\t".join(list(map(str, embed))) + "\n")

    print(
        "Ready! Go to http://projector.tensorflow.org/ and upload the files in the "
        "`projector` directory to visualize"
    )
