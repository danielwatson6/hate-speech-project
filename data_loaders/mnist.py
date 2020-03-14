import os

import pandas as pd
import tensorflow as tf

import boilerplate as tfbp

import tensorflow_datasets as tfds


@tfbp.default_export
class MNIST(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
    }

    def call(self):
        mnist_builder = tfds.builder("mnist")
        mnist_builder.download_and_prepare()
        datasets = mnist_builder.as_dataset()

        # And then the rest of your input pipeline
        train_dataset = self._transform_dataset(datasets["train"])
        test_dataset = self._transform_dataset(datasets["test"])

        return train_dataset, test_dataset

    def _transform_batch(self, batch):
        batch = tf.cast(batch["image"], tf.float32) / 255.0
        return tf.reshape(batch, [-1, 28 * 28])

    def _transform_dataset(self, dataset):
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(self._transform_batch)
        return dataset.prefetch(1)
