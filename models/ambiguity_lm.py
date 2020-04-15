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
        self.step = tf.Variable(0, trainable=False)
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

        # TODO: set weight tying as a boolean hyperparameter
        self.forward.add(tfkl.Dense(300))
        self.forward.add(tfkl.Lambda(lambda x: x @ tf.transpose(self.embed.weights[0])))

    def call(self, x):
        return self.forward(self.embed(self.word_to_id(x)))

    def loss(self, x):
        labels = self.word_to_id(x[:, 1:])
        logits = self(x[:, :-1])
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
            return final_ce + self.hparams.l2_penalty * l2_loss
        return final_ce

    @tfbp.runnable
    def fit(self, data_loader):
        opt = tf.optimizers.Adam(
            self.hparams.learning_rate,
            beta_1=self.hparams.beta_1,
            clipnorm=self.hparams.max_grad_norm,
        )

        # Train/validation split.
        train_dataset, valid_dataset = data_loader()
        valid_dataset_infinite = utils.infinite(valid_dataset)

        # Initialize the embedding matrix after building the model.
        if self.step.numpy() == 0:
            self(next(valid_dataset_infinite))
            utils.initialize_embeds(self.embed, data_loader.embedding_matrix)

        # TensorBoard writers.
        train_writer = self.make_summary_writer("train")
        valid_writer = self.make_summary_writer("valid")

        while self.epoch.numpy() < self.hparams.epochs:
            for batch in train_dataset:

                with tf.GradientTape() as g:
                    train_loss = tf.reduce_mean(self.loss(batch))
                grads = g.gradient(train_loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                step = self.step.numpy()
                if step % 100 == 0:
                    t0 = time()
                    valid_batch = next(valid_dataset_infinite)
                    valid_losses = self.loss(valid_batch)
                    valid_loss = tf.reduce_mean(valid_losses)
                    print(
                        "Step {} ({:.4f}s, train_loss={:.4f} valid_loss={:.4f})".format(
                            step, time() - t0, train_loss, valid_loss
                        ),
                        flush=True,
                    )
                    valid_ppx = tf.reduce_mean(tf.math.exp(valid_losses))

                    with train_writer.as_default():
                        tf.summary.scalar("loss", train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar("loss", valid_loss, step=step)
                        tf.summary.scalar("perplexity", valid_ppx, step=step)

                self.step.assign_add(1)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)
            self.save()

    @tfbp.runnable
    def ambiguity(self, data_loader):
        for x in data_loader():
            probs = self(x[:, :-1])
            # nlog_probs = -tf.math.log(probs + 1e-8)
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
            ppxs = tf.math.exp(self.loss(batch))
            for ppx in ppxs:
                total_ppx.append(ppx)
        print("{:.4f}".format(sum(total_ppx) / len(total_ppx)))

    @tfbp.runnable
    def evaluate(self, data_loader):
        self._evaluate(data_loader())
