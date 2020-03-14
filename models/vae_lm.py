import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp

import matplotlib.pyplot as plt
import imageio

from IPython import display
from models import utils


@tfbp.default_export
class VAE(tfbp.Model):
    default_hparams = {
        "batch_size": 32,
        "hidden_sizes": [512, 512],
        "latent_size": 2,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "epochs": 10,
        "num_layers": 2,
        "num_examples_to_generate": 16,
    }

    # Note: it is common to avoid using batch normalization when training VAEs
    # dense layer means that a layer receives input from all previous layers

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)

        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        for size in self.hparams.hidden_sizes:
            self.encoder.add(tfkl.Dense(size, activation=tf.nn.tanh))
            self.decoder.add(tfkl.Dense(size, activation=tf.nn.tanh))

        self.encoder.add(tfkl.Dense(self.hparams.latent_size * 2))
        self.decoder.add(tfkl.Dense(28 * 28))

        # tensorflow tutorial has this input layer...tf.keras.layers.InputLayer(input_shape=(latent_dim,))

        # for size in self.hparams.hidden_sizes:
        #     self.decoder.add(tfkl.Dense(size))
        # self.decoder.add(tfkl.Dense(28 * 28, activation=tf.nn.sigmoid))

    def sample(self, s=None):  # samples from a Gaussian distribution
        if s is None:
            s = tf.random.normal(shape=(100, self.hparams.latent_size))
        return self.decode(s, sigmoid=True)

    def encode(self, x):
        encoded = self.encoder(x)
        mean, var = tf.split(encoded, num_or_size_splits=2, axis=1)
        return mean, var

    def decode(self, z, sigmoid=False):
        logits = self.decoder(z)
        if sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs

        return logits

    # samples from Gaussian, multiplies by standard deviation and adds mean
    def reparameterize(self, mean, var):
        s = tf.random.normal(shape=mean.shape)
        return s * tf.exp(var * 0.5) + mean

    def kl_to_std_normal(self, mean, logvar):
        return 0.5 * tf.reduce_sum(-(logvar + 1.0) + tf.math.exp(logvar) + mean ** 2)

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        tf.debugging.assert_all_finite(mean, "mean")
        tf.debugging.assert_all_finite(logvar, "logvar")
        tf.debugging.assert_all_finite(z, "z")
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=1)
        kl = self.kl_to_std_normal(mean, logvar)

        return tf.reduce_mean(-logpx_z + kl)

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tfbp.runnable
    def fit(self, data_loader):
        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        # random_vector_for_generation = tf.random.normal(
        #     shape=[self.hparams.num_examples_to_generate, self.hparams.latent_size]
        # )

        train_data, valid_data = data_loader()
        valid_dataset_infinite = utils.infinite(valid_data)
        optimizer = tf.keras.optimizers.Adam(self.hparams.learning_rate)

        # Tensorboard writers
        train_writer = self.make_summary_writer("train")
        valid_writer = self.make_summary_writer("valid")

        while self.epoch.numpy() < self.hparams.epochs:
            for batch in train_data:
                self.compute_apply_gradients(batch, optimizer)
                train_loss = self.compute_loss(batch)
                step = self.step.numpy()
                if step % 100 == 0:
                    valid_batch = next(valid_dataset_infinite)
                    valid_loss = self.compute_loss(valid_batch)
                    print(
                        "Step {} (train_loss={:.4f} valid_loss={:.4f})".format(
                            step, train_loss.numpy(), valid_loss.numpy()
                        ),
                        flush=True,
                    )
                    with train_writer.as_default():
                        tf.summary.scalar("loss", train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar("loss", valid_loss, step=step)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)
            self.save()

    def _evaluate(self, dataset):
        total_ppx = []
        for batch in dataset:
            ppxs = tf.math.exp(self.compute_loss(batch))
            for ppx in ppxs:
                total_ppx.append(ppx)
        print("{:.4f}".format(sum(total_ppx) / len(total_ppx)))

    @tfbp.runnable
    def evaluate(self, data_loader):
        self._evaluate(data_loader())

