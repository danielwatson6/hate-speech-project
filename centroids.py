import os.path

import numpy as np
from scipy.spatial.distance import cosine

import utils


if __name__ == "__main__":
    w2v = utils.load_word2vec()
    avg_embed = lambda vs: sum(vs) / 300

    stormfront_vals = [[], []]
    for tokens, label in utils.stormfront_gen():
        for token in tokens:
            if token in w2v:
                stormfront_vals[label].append(w2v[token])
    stormfront_vals = list(map(avg_embed, stormfront_vals))

    twitter_vals = [[], []]
    for tokens, label in utils.twitter_gen():
        for token in tokens:
            if token in w2v:
                twitter_vals[label].append(w2v[token])
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
    print()

    print("Normalized distances")
    ndist = lambda x, y: np.linalg.norm(x / np.linalg.norm(x) - y / np.linalg.norm(y))
    print("||sn - sh|| =", ndist(sn, sh))
    print("||tn - th|| =", ndist(tn, th))
    print("||sn - tn|| =", ndist(sn, tn))
    print("||sh - th|| =", ndist(sh, th))
    print()

    print("Cosine distances")
    print("cos_dist(sn, sh)", cosine(sn, sh))
    print("cos_dist(tn, th)", cosine(tn, th))
    print("cos_dist(sn, tn)", cosine(sn, tn))
    print("cos_dist(sh, th)", cosine(sh, th))
