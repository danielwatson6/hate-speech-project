import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn


if __name__ == "__main__":

    frequencies = {}
    for word in wn.all_lemma_names():
        n = len(word)
        if n not in frequencies:
            frequencies[n] = 0
        frequencies[n] += 1

    bins = []
    counts = []
    for f, count in frequencies.items():
        bins.append(f)
        counts.append(count)

    plt.bar(bins, counts)
    plt.show()
