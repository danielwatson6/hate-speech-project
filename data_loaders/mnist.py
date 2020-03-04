from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

import boilerplate as tfbp


@tfbp.default_export
class MNIST(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
    }

    def call(self):
        mnist_builder = tfds.builder("mnist")
        mnist_builder.download_and_prepare()
        ds_train, ds_test = mnist_builder.as_dataset(split=["train", "test"])
        ds_train = ds_train.shuffle(1024).batch(self.hparams.batch_size)
        ds_test = ds_test.shuffle(1024).batch(self.hparams.batch_size)
        ds_train = ds_train.prefetch(1)
        ds_test = ds_test.prefetch(1)
        return ds_train, ds_test

