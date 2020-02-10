import os

import tensorflow as tf

import boilerplate as tfbp
from data_loaders import utils


@tfbp.default_export
class WikiText(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "corpus": 2,
        "max_seq_len": 40,
        "punctuation": True,
        "lowercase": True,
    }

    def call(self):
        if self.hparams.corpus == 103:
            self._data_path = os.path.join("data", "wikitext-103-raw")
        elif self.hparams.corpus == 2:
            self._data_path = os.path.join("data", "wikitext-2")
        else:
            raise ValueError("`corpus` hyperparameter can only attain values 2 or 103.")

        vocab_path = os.path.join(data_path, "wiki.vocab")
        embeds_path = os.path.join(data_path, "wiki.vocab")

        # Used by models to display outputs as strings; the conversion is data-dependent.
        self.word_to_id, self.id_to_word = utils.make_word_id_maps(
            vocab_path, self.hparams.vocab_size
        )

        # Used by models to find the initial values of the embedding matrix, which are
        # data-dependent.
        self.embedding_matrix = utils.save_or_load_embeds(
            embeds_path, vocab_path, self.hparams.vocab_size
        )

    def call(self):
        if self.method == "fit":
            train_dataset = self._make_dataset(
                os.path.join(data_path, "wiki.train.clean"), shuffle=10000
            )
            valid_dataset = self._make_dataset(
                os.path.join(data_path, "wiki.valid.clean"), shuffle=10000
            )
            return train_dataset, valid_dataset

        elif self.method == "evaluate":
            return self._make_dataset(os.path.join(data_path, "wiki.test.clean"))

        elif self.method == "interact":

            def interact_mode_generator():
                while True:
                    yield [input("Type a sentence: ")]

            dataset = tf.data.Dataset.from_generator(interact_mode_generator, tf.string)
            return dataset.map(self._batch_to_ids)

    def _batch_to_ids(self, batch):
        sequences = tf.strings.split(batch)
        # TODO: implement optional lowercasing and punctuation removal.
        if self.hparams.lowercase:
            ...
        if not self.hparams.punctuation:
            ...

        # Convert the ragged tensor to a regular tensor. This takes care of padding.
        padded = sequences.to_tensor(default_value="<pad>")
        if self.hparams.max_seq_len:
            padded = padded[:, : self.hparams.max_seq_len]
        return self.word_to_id(padded)

    def _make_dataset(self, path, shuffle=None):
        dataset = tf.data.TextLineDataset(path)

        if shuffle:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(
            self._batch_to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return dataset.prefetch(1)
