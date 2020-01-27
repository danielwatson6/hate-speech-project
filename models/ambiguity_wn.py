"""Wordnet baseline for measuring ambiguity."""

from nltk.corpus import wordnet as wn

import boilerplate as tfbp


@tfbp.default_export
class AmbiguityWordnet(tfbp.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tfbp.runnable
    def ambiguity(self, data_loader):
        dataset = data_loader()
        for batch in dataset:
            batch = data_loader.id_to_word(batch)
            for seq in batch.numpy():
                scores = []
                for word in seq:
                    word = word.decode("utf8")
                    score = len(wn.synsets(word))
                    scores.append(score)
                print(", ".join(["{:.4f}".format(x) for x in scores]))
