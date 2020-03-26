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
                    tfkl.LSTM(hs, dropout=self.hparams.dropout, return_sequences=True)
                )
            )
        self.encoder.add(tfkl.Dense(600))

        for hs in self.hparams.dec_hidden:
            self.decoder.add(
                tfkl.LSTM(hs, dropout=self.hparams.dropout, return_sequences=True)
            )

        self.decoder.add(tfkl.Dense(self.hparams.vocab_size))

    def encode(self, x):
        rnn_outputs = self.encoder(x)
        mean, log_var = tf.split(rnn_outputs, num_or_size_splits=2, axis=2)
        return mean, log_var

    def decode(self, rnn_inputs, softmax=False):
        # = E_{z~q(|)} [
        #   log p(x_2|z,x_1) + log p(x_3|z,x_1,x_2) + ... + log p(x_T|z,x_1,...,x_T-1)
        # + log p(x_3|z,x_2) ... + log p(x_T|z,x_2,...,x_T-1) + log p(x_1|z,x2,...,x_T)
        # ...
        # + log p(x_T|z,x_T-1) + log p(x_1|z,x_T-1,x_T) + ... + log p(x_T-2|z,x_T-1,xT,...,x_T-3)
        # + log p(x_1|z,x_T) + log p(x_2|z,x_T,x_1) + ... + log p(x_T-1|z,x_T,x1,x_T-2)
        # ]
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
        return 0.5 * tf.reduce_sum(
            lv2
            - lv1
            - 1.0
            + tf.math.exp(lv1 - lv2)
            + tf.math.exp(-lv2) * (m1 - m2) ** 2,
            axis=-1,
        )

    # @tf.function
    def compute_loss(self, x):
        batch_size = tf.shape(mean)[0]
        eos_tokens = tf.tile([3], [batch_size, 1])
        x = tf.concat([x, eos_tokens], 1)

        x_embedded = self.embed(x)
        mean, logvar = self.encode(x_embedded)
        z = self.reparameterize(mean, logvar)

        # TODO
        seq_lengths = tf.reduce_sum(tf.cast(tf.not_equal(x, 0), tf.float32), axis=1)

        total_re = 0.0
        for i, T in enumerate(seq_lengths):
            inputs = []
            labels = []
            for j in range(T):
                input_ = []
                label = []
                for k in range(T - 1):
                    input_.append(tf.concat([x_embedded[i, (j + k) % T], z[i, j]], 0))
                    label.append(x[i, (j + k + 1) % T])
                inputs.append(input_)
                labels.append(label)

            logits = self.decoder(inputs)
            re = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            )
            # TODO: reduce as running average.
            total_re += tf.reduce_sum(re)

        kl = self.kl_gaussian(
            [mean, logvar], [x_embedded, tf.math.log(self.hparams.prior_variance)]
        )

        return total_re, tf.reduce_mean(kl)

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
