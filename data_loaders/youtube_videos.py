import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class YouTubeVideos(tfbp.DataLoader):
    default_hparams = {"batch_size": 32, "corpus": 2, "max_seq_len": 40}

    def call(self):
        if self.hparams.corpus == 2:
            vocab_path = os.path.join("data", "wikitext-103-raw")
        elif self.hparams.corpus == 103:
            vocab_path = os.path.join("data", "wikitext-2")
        else:
            raise ValueError("`corpus` hyperparameter can only attain values 2 or 103.")

        vocab_path = os.path.join(vocab_path, "wiki.vocab.tsv")

        # Args: filename, key_dtype, key_index, value_dtype, value_index, vocab_size
        word_to_id_init = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.string,
            0,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            vocab_size=self.hparams.vocab_size,
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

        if self.method == "segment":
            dataset = tf.data.Dataset.from_generator(self._get_generator(), tf.string)
            return self._transform_dataset(dataset)

    def _get_generator(self):
        def _g():
            data_path = os.path.join(os.environ["DATASETS"], "youtube_right")
            for filename in os.listdir(data_path):
                df = pd.read_csv(os.path.join(data_path, filename))
                for row in df.iterrows():
                    transcript = re.sub(r"[0-9]+:[0-9]+", " ", row[0]["content"])
                    transcript = re.sub(r"\n", " ", transcript.lower().strip())
                    transcript = transcript.split()
                    i = 0
                    batch = []
                    while i < len(transcript):
                        batch.append(transcript[i : i + self.hparams.max_seq_len])
                        i += self.hparams.max_seq_len
                    yield batch

        return _g

    def _batch_to_ids(self, batch):
        padded = tf.strings.split(batch + " <eos>").to_tensor(default_value="<pad>")
        if self.hparams.max_seq_len:
            padded = padded[:, : self.hparams.max_seq_len]
        return self.word_to_id.lookup(padded)

    def _transform_dataset(self, dataset):
        dataset = dataset.map(self._batch_to_ids)
        dataset = dataset.shuffle(10000)
        return dataset.prefetch(1)
