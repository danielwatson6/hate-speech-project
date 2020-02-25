import os

import tensorflow as tf

import boilerplate as tfbp
from data_loaders import utils


@tfbp.default_export
class WikiText(tfbp.DataLoader):
    default_hparams = {
        "vocab_size": 20000,
        "batch_size": 32,
        "corpus": 2,
        "max_seq_len": 40,
        "punctuation": True,
        "lowercase": True,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self.hparams.corpus == 103:
            self._data_path = os.path.join("data", "wikitext-103-raw")
        elif self.hparams.corpus == 2:
            self._data_path = os.path.join("data", "wikitext-2")
        else:
            raise ValueError("`corpus` hyperparameter can only attain values 2 or 103.")

        vocab_path = os.path.join(self._data_path, "wiki.vocab.tsv")
        embeds_path = os.path.join(self._data_path, "wiki.npy")

        # Used by models to find the initial values of the embedding matrix, which are
        # data-dependent.
        # TODO: what if the model doesn't need an embedding matrix? Waste of memory.
        self.embedding_matrix = utils.save_or_load_embeds(
            embeds_path, vocab_path, self.hparams.vocab_size
        )

    def call(self):
        if self.method == "fit":
            train_dataset = self._make_dataset("wiki.train.clean", shuffle=10000)
            valid_dataset = self._make_dataset("wiki.valid.clean", shuffle=10000)
            return train_dataset, valid_dataset

        elif self.method == "eval_test":
            return self._make_dataset("wiki.test.clean")

        return self._make_dataset("wiki.valid.clean")

    def _preprocess(self, batch):
        if not self.hparams.punctuation:
            batch = tf.strings.regex_replace(batch, "[\.,;:-]", "")
        if self.hparams.lowercase:
            batch = tf.strings.lower(batch)

        # No need to shrink spaces, this is handled correctly by `tf.strings.split`.
        padded = tf.strings.split(batch).to_tensor(default_value="<pad>")

        if self.hparams.max_seq_len:
            padded = padded[:, : self.hparams.max_seq_len]
        return padded

    def _make_dataset(self, filename, shuffle=None):
        dataset = tf.data.TextLineDataset(os.path.join(self._data_path, filename))

        if shuffle:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(
            self._preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return dataset.prefetch(1)
