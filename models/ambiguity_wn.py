"""Wordnet baseline for measuring ambiguity."""

from nltk.corpus import wordnet as wn

import boilerplate as tfbp


@tfbp.default_export
class AmbiguityWordnet(tfbp.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tfbp.runnable
    def ambiguity(self, data_loader):
        ...
