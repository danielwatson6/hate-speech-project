"""Twitter moral foundations data loader."""

from collections import Counter
import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MF(tfbp.DataLoader):
    default_hparams = {"batch_size": 32, "vocab_size": 20000}

    def load(self):
        vocab_path = os.path.join("data", "twitter_mf.clean.vocab")
        data_path = os.path.join("data", "twitter_mf.clean.shuffled.csv")

        if not os.path.isfile(vocab_path):
            counter = Counter()

            df = pd.read_csv(data_path)
            for row in df.iterrows():
                tokens = str(row[1]["tweet"]).strip().split()
                for token in tokens:
                    if token not in counter:
                        counter[token] = 0
                    counter[token] += 1

            with open(vocab_path, "w") as f:
                f.write("<PAD>\n")
                f.write("<UNK>\n")
                f.write("<SOS>\n")
                f.write("<EOS>\n")

                for word, _ in counter.most_common():
                    f.write(word + "\n")

        # filename, key_dtype, key_index, value_dtype, value_index, vocab_size
        table_initializer = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.string,
            tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            vocab_size=self.hparams.vocab_size,
        )
        self._word_to_id = tf.lookup.StaticHashTable(table_initializer, 1)

        dataset = tf.data.experimental.make_csv_dataset(
            data_path,
            self.hparams.batch_size,
            num_epochs=1,
            shuffle=False,
            num_rows_for_inference=None,
        )
        dataset = dataset.map(self.dict_to_pair)
        return dataset.prefetch(1)

    def dict_to_pair(self, batch):
        # tweets = tf.strings.unicode_decode(batch["tweet"], input_encoding="UTF-8")
        tweets = tf.strings.split(batch["tweet"]).to_tensor(default_value="<PAD>")
        tweets = self._word_to_id.lookup(tweets)

        labels = [batch[k] for k in batch.keys() if k not in ["id", "tweet"]]
        labels = tf.stack(labels)
        labels = tf.cast(labels, tf.float32)
        labels = tf.transpose(labels)

        return tweets, labels
