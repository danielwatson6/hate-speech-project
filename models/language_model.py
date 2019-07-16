"""Word-level seq2seq model with attention."""

import os
import re

from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp


def _seq_to_str(seq, id_to_word):
    seq = id_to_word(seq).numpy()
    seq = " ".join([token.decode("utf-8") for token in seq])
    return re.sub(r" <eos>.*", "", seq)


@tfbp.default_export
class LanguageModel(tfbp.Model):
    default_hparams = {
        "rnn_layers": 2,
        "batch_size": 32,
        "vocab_size": 20000,
        "hidden_size": 512,
        "optimizer": "adam",  # "sgd" or "adam"
        "learning_rate": 0.001,
        "epochs": 5,
        "dropout": 0.0,
        "corpus": 2,  # 2 or 103
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)

        self.embed = self._make_embed()

        dropout = 0.0
        if self.method == "fit":
            dropout = self.hparams.dropout

        self.encoder = tf.keras.Sequential()
        for _ in range(self.hparams.rnn_layers):
            self.encoder.add(
                tfkl.GRU(
                    self.hparams.hidden_size, dropout=dropout, return_sequences=True
                )
            )
        self.encoder.add(tfkl.TimeDistributed(tfkl.Dense(self.hparams.vocab_size)))

    def _make_embed(self):
        # Embedding matrix. TODO: move data-dependent stuff to data loader.
        word2vec = KeyedVectors.load(os.path.join("data", "word2vec"), mmap="r")
        embedding_matrix = np.random.uniform(
            low=-1.0, high=1.0, size=(self.hparams.vocab_size, 300)
        )
        corpus = self.hparams.corpus
        if corpus == 103:
            corpus = str(corpus) + "-raw"
        with open(os.path.join("data", f"wikitext-{corpus}", "wiki.vocab.tsv")) as f:
            for i, word in enumerate(f):
                word = word.strip()
                if word in word2vec:
                    embedding_matrix[i] = word2vec[word]
        return tfkl.Embedding(
            self.hparams.vocab_size,
            300,
            embeddings_initializer=tf.initializers.constant(embedding_matrix),
        )

    def _loss(self, y_true, y_pred):
        mask = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
        seq_lengths = tf.expand_dims(tf.reduce_sum(mask, axis=1), 1)
        ce = tf.losses.SparseCategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
        result = ce(y_true, y_pred, sample_weight=mask)
        return tf.reduce_mean(result / seq_lengths)

    def call(self, y):
        # Encoder initial state and input.
        sos_ids = tf.cast([[2]] * y.shape[0], tf.int64)
        h0 = tf.zeros([-1, self.hparams.rnn_layers, self.hparams.hidden_size])
        h0 = tf.unstack(dec_st, self.hparams.rnn_layers, axis=1)

        x = self.embed(tf.concat([sos_ids, y], 1))
        return self.encoder(x)

    def fit(self, data_loader):
        """Method invoked by `run.py`."""

        # Optimizer.
        if self.hparams.optimizer == "adam":
            opt = tf.optimizers.Adam(self.hparams.learning_rate)
        else:
            opt = tf.optimizers.SGD(self.hparams.learning_rate)

        # Train/validation split. Keep a copy of the original validation data to
        # evalute at the end of every epoch without falling to an infinite loop.
        train_dataset, valid_dataset_orig = data_loader()
        valid_dataset = iter(valid_dataset_orig.repeat())

        # TensorBoard writers.
        train_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "train")
        )
        valid_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "valid")
        )

        max_eval_score = float("-inf")

        while self.epoch.numpy() < self.hparams.epochs:
            for y in train_dataset:

                # TODO: try to remove this
                if not self.built:
                    self(y)

                with tf.GradientTape() as g:
                    train_probs = self(y)
                    train_loss = self._loss(y, train_probs)

                grads = g.gradient(train_loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                step = self.step.numpy()
                if step % 100 == 0:
                    y = next(valid_dataset)
                    valid_probs = self(y)
                    valid_loss = self._loss(y, valid_probs)
                    valid_output = tf.math.argmax(valid_probs[0], axis=-1)

                    with train_writer.as_default():
                        tf.summary.scalar("perplexity", tf.exp(train_loss), step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar("perplexity", tf.exp(valid_loss), step=step)

                    print("Sample validation input:")
                    print(_seq_to_str(y[0], data_loader.id_to_word.lookup))
                    print("Sample validation output:")
                    print(_seq_to_str(valid_output, data_loader.id_to_word.lookup))

                if step % 1000 == 0:
                    self.save()

                print("Step {} (train_loss={:.4f})".format(step, train_loss))
                self.step.assign_add(1)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)
            self.save()

    def ambiguity(self, data_loader):
        """Method invoked by `run.py`."""
        dataset = data_loader()

        print("sentence_length,sentence_ppx,main_word,main_word_ppx")
        for batch_in in data_loader:
            batch_out = self(batch_in)
            batch_loss = self._loss(batch_in, batch_out)

            for y, probs in zip(batch_in, batch_out):
                sentence_length = len(_seq_to_str(y).split())
                sentence_ppx = tf.math.exp(
                    tf.reduce_sum(tf.math.log(tf.reduce_max(probs, axis=-1)))
                )
                main_word = 0  # TODO: replace with proper id
                main_word_ppx = tf.reduce_max(probs[main_word])

                print(f"{sentence_length},{sentence_ppx},{main_word},{main_word_ppx}\n")

    def _evaluate(self, dataset):
        """Perplexity evaluation.

        Kept as a separate method in case we want to evaluate durining training (e.g.
        per-epoch evaluations to allow remembering the optimal stopping point).

        """
        ppx = []
        for batch_in in dataset:
            batch_out = self(batch_in)
            for y, probs in zip(batch_in, batch_in):
                ppx.append(tf.exp(self._loss(y, probs)))
        return sum(ppx) / len(ppx)

    def evaluate(self, data_loader):
        """Method invoked by `run.py`."""
        dataset = data_loader()
        print(self._evaluate(dataset))
