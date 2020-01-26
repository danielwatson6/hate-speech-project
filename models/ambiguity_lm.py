"""Language model for measuring ambiguity."""

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import utils


@tfbp.default_export
class LM(tfbp.Model):
    default_hparams = {
        "batch_size": 32,
        "vocab_size": 20000,
        "hidden_sizes": [512, 512],
        "dropout": 0.0,
        "fine_tune_embeds": False,
        "use_lstm": True,  # GRU will be used if set to false.
        "learning_rate": 1e-3,
        "epochs": 10,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)

        self.embed = tfkl.Embedding(self.hparams.vocab_size, 300)
        self.embed.trainable = self.hparams.fine_tune_embeds

        self.forward = tfkl.Sequential()

        if self.hparams.use_lstm:
            RNN = tfkl.LSTM
        else:
            RNN = tfkl.GRU

        dropout = 0.0
        if self.method == "train":
            dropout = self.hparams.dropout

        for size in self.hparams.hidden_sizes:
            self.forward.add(RNN(size, dropout=dropout, return_sequences=True))

        self.forward.add(tfkl.Dense(self.hparams.vocab_size), activation=tf.nn.softmax)

        self.cross_entropy = tf.losses.SparseCategoricalCrossentropy()

    def call(self, x):
        return self.forward(self.embed(x))

    def loss(self, x):
        probs = self(x[:-1])
        # TODO: mask the loss.
        return self.cross_entropy(x[1:], probs)

    @tfbp.runnable
    def fit(self, data_loader):
        opt = tf.optimizers.Adam(self.hparams.learning_rate)

        # Initialize the embedding matrix.
        if self.step.numpy() == 0:
            utils.initialize_embeds(self.embeds, data_loader.embedding_matrix)

        # Train/validation split.
        train_dataset, valid_dataset = data_loader()
        valid_dataset_infinite = utils.infinite(valid_dataset)

        # TensorBoard writers.
        train_writer = self.make_summary_writer("train")
        valid_writer = self.make_summary_writer("valid")

        while self.epoch.numpy() < self.hparams.epochs:
            for x in train_dataset:
                opt.minimize(self.loss(x), self.trainable_weights)

                step = self.step.numpy()
                if step % 100 == 0:
                    x = next(valid_dataset_infinite)
                    valid_loss = self.loss(x)
                    print(
                        "Step {} (train_loss={:.4f} valid_loss={:.4f})".format(
                            step, train_loss, valid_loss
                        ),
                        flush=True,
                    )

                    with train_writer.as_default():
                        tf.summary.scalar("loss", train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar("loss", valid_loss, step=step)

                self.step.assign_add(1)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)
            self.save()

    @tfbp.runnable
    def ambiguity(self, data_loader):
        # TODO: implement this. Iterate over all the batches in the data loader, and
        # print the ambiguity score per example for each word. From the softmax outputs
        # [prob_word1, ..., prob_wordV], this can be computed as
        # - prob_word1 log prob_word1 - ... - prob_wordV log prob_wordV
        ...

    def _evaluate(self, dataset):
        # TODO: research and implement a standard benchmark used to evaluate language
        # models (BLEU score??)
        ...

    @tfbp.runnable
    def evaluate(self, data_loader):
        self._evaluate(data_loader())
