import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import utils


class ImpliednessVAE(tfbp.Model):
    default_hparams = {
        "vocab_size": 20000,
        "enc_hidden": [512, 512],
        "dec_hidden": [128, 128],
        "dropout": 0.0,
        "opt": "adam",
        "lr": 1e-3,
        "prior_variance": 0.1,
    }

    def __init__(self, *a, **kw):
        super().__init(*a, **kw)

        if self.hparams.opt.lower() == "adam":
            self.opt = tf.optimizers.Adam(self.hparams.lr)
        else:
            self.opt = tf.optimizers.SGD(self.hparams.lr)

        self.embed = tfkl.Embedding(self.hparams.vocab_size, 300)
        self.embed.trainable = False

        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()

        for hs in self.hparams.enc_hidden:
            self.encoder.add(
                tfkl.Bidirectional(
                    tfkl.LSMT(hs, dropout=self.hparams.dropout, return_sequence=True)
                )
            )
        self.encoder.add(tfkl.Dense(600))

        for hs in self.hparams.dec_hidden:
            self.decoder.add(
                tfkl.LSMT(hs, dropout=self.hparams.dropout, return_sequence=True)
            )

        self.decoder.add(tfkl.Dense(self.hparams.vocab_size))

    def encode(self, x):
        rnn_outputs = self.encoder(x)
        mean, log_var = tf.split(rnn_outputs, num_or_size_splits=2, axis=2)
        return mean, log_var

    def decode(self, rnn_inputs, softmax=False):
        logits = self.decoder(rnn_inputs)
        if softmax:
            return tf.nn.softmax(logits)
        return logits

    def reparametrize(self, mean, logvar):
        shape = tf.random.normal(shape=tf.shape(mean))
        return shape * tf.exp(logvar * 0.5) + mean

    def kl_gaussian(self, p, q):
        """Compute KL[p||q] where p and q are (mean, log_var) tuples."""
        m1, lv1 = p
        m2, lv2 = q
        # TODO: figure out how to reduce this.
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        return 0.5 * (lv2 - lv1 - 1.0) + (
            tf.math.exp(lv1) * (m1 - m2) ** 2
        ) * tf.math.reciprocal(2 * tf.math.exp(lv2))

    def compute_loss(self, x):
        x_embedded = self.embed(x)
        mean, logvar = self.encode(x_embedded)
        z = self.reparameterize(mean, logvar)

        batch_size = tf.shape(mean)[0]
        sos_tokens = tf.tile([[self.embed(2)]], [batch_size, 1, 1])
        x_shifted = tf.concat([sos_tokens, x_embedded[:, :-1]], 1)
        rnn_inputs = tf.concat([x_shifted, z], 2)

        x_logit = self.decode(z, rnn_inputs)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=1)
        # TODO
        kl = self.kl_to_std_normal(mean, logvar)

        return tf.reduce_mean(-logpx_z), tf.reduce_mean(kl)

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            re, kl = self.compute_loss(x)
            loss = re + kl
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tfbp.runnable
    def fit(self, data_loader):
        train_data, valid_data = data_loader()
        valid_dataset_infinite = utils.infinite(valid_data)

        # Tensorboard writers
        train_writer = self.make_summary_writer("train")
        valid_writer = self.make_summary_writer("valid")
        ...
