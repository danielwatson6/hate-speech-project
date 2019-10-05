"""Twitter moral foundations data loader."""

from collections import Counter
import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MF(tfbp.DataLoader):
    default_hparams = {"batch_size": 32, "vocab_size": 25047}

    def call(self):
        vocab_path = os.path.join("data", "twitter_mf.clean.vocab")
        data_path = os.path.join("data", "twitter_mf.clean.shuffled.csv")

        if not os.path.isfile(vocab_path):
            counter = Counter()

            df = pd.read_csv(data_path)
            for i, row in enumerate(df.iterrows()):
                if i < 4771:
                    continue
                tokens = str(row[1]["tweet"]).strip().split()
                for token in tokens:
                    if token not in counter:
                        counter[token] = 0
                    counter[token] += 1

            with open(vocab_path, "w") as f:
                f.write("<pad>\n")
                f.write("<unk>\n")

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
        dataset = dataset.map(self._dict_to_pair)
        return dataset.prefetch(1)

    def _dict_to_pair(self, batch):
        tweets = batch["tweet"]
        tweets = tf.strings.split(tweets).to_tensor(default_value="<pad>")
        tweets = self._word_to_id.lookup(tweets)

        labels = [
            batch[k]
            for k in ["authority", "fairness", "care", "loyalty", "purity", "non_moral"]
        ]
        labels = tf.stack(labels)
        labels = tf.cast(labels, tf.float32)
        labels = tf.transpose(labels)

        return tweets, labels
