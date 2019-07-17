import os

import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class WikiText(tfbp.DataLoader):
    default_hparams = {"batch_size": 32, "corpus": 2, "max_seq_len": 40}

    def call(self):
        if self.hparams.corpus == 103:
            data_path = os.path.join("data", "wikitext-103-raw")
        elif self.hparams.corpus == 2:
            data_path = os.path.join("data", "wikitext-2")
        else:
            raise ValueError("`corpus` hyperparameter can only attain values 2 or 103.")

        vocab_path = os.path.join(data_path, "wiki.vocab.tsv")

        # Args: filename, key_dtype, key_index, value_dtype, value_index, vocab_size
        word_to_id_init = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.string,
            0,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            vocab_size=self.hparams.vcab_size,
        )
        id_to_word_init = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            tf.string,
            0,
            vocab_size=self.hparams.vocab_size,
        )
        self.word_to_id = tf.lookup.StaticHashTable(word_to_id_init, 1)
        self.id_to_word = tf.lookup.StaticHashTable(id_to_word_init, "<unk>")

        if self.method == "fit":
            train_inputs = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.train.inputs")
            )
            train_labels = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.train.labels")
            )
            valid_inputs = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.valid.inputs")
            )
            valid_labels = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.valid.labels")
            )
            train_dataset = tf.data.Dataset.zip((train_inputs, train_labels))
            valid_dataset = tf.data.Dataset.zip((valid_inputs, valid_labels))

            train_dataset = self._transform_dataset(train_dataset)
            valid_dataset = self._transform_dataset(valid_dataset)

            return train_dataset, valid_dataset

        elif self.method == "evaluate":
            test_inputs = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.test.inputs")
            )
            test_labels = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.test.labels")
            )
            test_dataset = tf.data.Dataset.zip((test_inputs, test_labels))

            return self._transform_dataset(test_dataset)

        elif self.method == "interact":

            def interact_mode_generator():
                while True:
                    yield [input("Type a sentence: ")]

            dataset = tf.data.Dataset.from_generator(interact_mode_generator, tf.string)
            return dataset.map(self._batch_to_ids)

    def _batch_to_ids(self, batch):
        padded = tf.strings.split(batch + " <eos>").to_tensor(default_value="<pad>")
        if self.hparams.max_seq_len:
            padded = padded[:, : self.hparams.max_seq_len]
        return self.word_to_id.lookup(padded)

    def _transform_dataset(self, dataset):
        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(
            lambda x, y: (self._batch_to_ids(x), self._batch_to_ids(y))
        )
        dataset = dataset.shuffle(10000)
        return dataset.prefetch(1)
