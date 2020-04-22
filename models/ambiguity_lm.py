"""Language model for measuring ambiguity."""

import os
from time import time

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
        "recurrent_dropout": 0.0,
        "fine_tune_embeds": False,
        "use_lstm": True,  # GRU will be used if set to false.
        "learning_rate": 1e-3,
        "beta_1": 0.9,
        "max_grad_norm": 10.0,
        "l2_penalty": 0.0,
        "epochs": 10,
        # TODO: find a way to make the model not use this. The hash tables for word<->id
        # conversion are immutable and cannot be overwritten as we do with the embedding
        # matrix.
        "vocab_path": "",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.epoch = tf.Variable(0, trainable=False)

        # TODO: find a way to make the model not use this.
        if not self.hparams.vocab_path:
            raise ValueError("Please specify --vocab_path=path/to/vocab.tsv")
        self.word_to_id, self.id_to_word = utils.make_word_id_maps(
            self.hparams.vocab_path, self.hparams.vocab_size,
        )

        self.embed = tfkl.Embedding(self.hparams.vocab_size, 300)
        self.embed.trainable = self.hparams.fine_tune_embeds

        self.forward = tf.keras.Sequential()

        if self.hparams.use_lstm:
            RNN = tfkl.LSTM
        else:
            RNN = tfkl.GRU

        dropout = 0.0
        recurrent_dropout = 0.0
        if self.method == "fit":
            dropout = self.hparams.dropout
            recurrent_dropout = self.hparams.recurrent_dropout

        for size in self.hparams.hidden_sizes:
            self.forward.add(
                RNN(
                    size,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=True,
                )
            )

        self.forward.add(tfkl.Dense(self.hparams.vocab_size))

        self.opt = tf.optimizers.Adam(
            self.hparams.learning_rate,
            beta_1=self.hparams.beta_1,
            clipnorm=self.hparams.max_grad_norm,
        )

    def call(self, x):
        return self.forward(self.embed(self.word_to_id(x)))

    def loss_and_output(self, x):
        labels = self.word_to_id(x[:, 1:])
        logits = self(x[:, :-1])

        output = self.id_to_word(tf.math.argmax(logits, axis=-1))

        # Avoid punishing the model for "wrong" guesses on padded data.
        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        masked_ce = ce * mask
        # Compute means by correct length, dropping any sequences of length 0.
        sequence_lengths = tf.reduce_sum(mask, axis=1)
        mean_factor = tf.map_fn(
            lambda x: tf.cond(x == 0.0, lambda: 0.0, lambda: 1.0 / x), sequence_lengths
        )
        final_ce = tf.reduce_sum(masked_ce, axis=1) * mean_factor
        if self.hparams.l2_penalty > 0:
            l2_loss = sum(tf.nn.l2_loss(v) for v in self.trainable_weights)
            return final_ce + self.hparams.l2_penalty * l2_loss, output
        return final_ce, output

    @tf.function
    def fit_epoch(self, train_dataset, valid_dataset):

        for batch in train_dataset:

            t0 = tf.timestamp()
            with tf.GradientTape() as g:
                losses, _ = self.loss_and_output(batch)
                train_loss = tf.reduce_mean(losses)
            grads = g.gradient(train_loss, self.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.trainable_weights))
            t1 = tf.timestamp()

            if self.step % 100 == 0:

                valid_batch = next(valid_dataset)
                valid_losses, valid_output = self.loss_and_output(valid_batch)
                valid_loss = tf.reduce_mean(valid_losses)
                valid_ppx = tf.reduce_mean(tf.math.exp(valid_losses))

                tf.print("step", self.step)
                tf.print("  train step time", t1 - t0)
                tf.print("  train_loss", train_loss)
                tf.print("  valid_loss", valid_loss)

                with self.train_writer.as_default():
                    tf.summary.scalar("loss", train_loss, step=self.step)
                with self.valid_writer.as_default():
                    tf.summary.scalar("loss", valid_loss, step=self.step)
                    tf.summary.scalar("perplexity", valid_ppx, step=self.step)

            self.step.assign_add(1)

    @tfbp.runnable
    def fit(self, data_loader):

        # Train/validation split.
        train_dataset, valid_dataset = data_loader()
        valid_dataset = utils.infinite(valid_dataset)

        # Initialize the embedding matrix after building the model.
        if self.step.numpy() == 0:
            self(next(valid_dataset))
            utils.initialize_embeds(self.embed, data_loader.embedding_matrix)

        # TensorBoard writers.
        self.train_writer = self.make_summary_writer("train")
        self.valid_writer = self.make_summary_writer("valid")

        while self.epoch < self.hparams.epochs:
            self.fit_epoch(train_dataset, valid_dataset)
            tf.print(f"Epoch {self.epoch} finished")
            self.epoch.assign_add(1)
            self.save()

    @tfbp.runnable
    def ambiguity(self, data_loader):
        for x in data_loader():
            probs = tf.nn.softmax(self(x[:, :-1]))
            nlog_probs = -tf.math.log(probs)
            nplogp = probs * nlog_probs

            mask = tf.cast(tf.not_equal(self.word_to_id(x[:, 1:]), 0), tf.float32)
            entropy = tf.reduce_sum(nplogp, axis=2).numpy() * mask
            # for loop to iterate and test
            for sequence in entropy:
                print(", ".join(["{:.4f}".format(y) for y in sequence]))

    def _evaluate(self, dataset):
        total_ppx = []
        for batch in dataset:
            losses, _ = self.loss_and_output(batch)
            ppxs = tf.math.exp(losses)
            for ppx in ppxs:
                total_ppx.append(ppx)
        print("{:.4f}".format(sum(total_ppx) / len(total_ppx)))

    @tfbp.runnable
    def evaluate(self, data_loader):
        self._evaluate(data_loader())
