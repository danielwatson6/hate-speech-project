"""Twitter moral foundations data loader."""

import os

import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MF(tfbp.DataLoader):
    default_hparams = {"batch_size": 32, "softmax_labels": False}

    def load(self):
        dataset = tf.data.experimental.make_csv_dataset(
            os.path.join("data", "twitter_mf.csv"), self.hparams.batch_size
        )
        dataset = dataset.map(self.dict_to_pair)
        return dataset.prefetch(1)

    def dict_to_pair(self, batch):
        labels = tf.stack(
            [batch[k] for k in batch.keys() if k not in ["index", "tweet"]]
        )
        labels = tf.cast(labels, tf.float32)
        labels = tf.transpose(labels)

        if self.hparams.softmax_labels:
            labels = tf.nn.softmax(labels)
        else:
            Z = tf.expand_dims(tf.reduce_sum(labels, axis=1), 1)
            labels /= Z

        return batch["tweet"], labels
