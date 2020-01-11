"""Language model."""

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
import utils


@tfbp.default_export
class LM(tfbp.Model):
    default_hparams = {
        "batch_size": 32,
        "vocab_size": 25047,  # all vocabulary
        "fine_tune_embeds": True,
        "learning_rate": 1e-3,
        "epochs": 10,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        embeds_path = os.path.join("data", "twitter_mf.clean.npy")
        embeds_path = os.path.join("data", "twitter_mf.clean.vocab")
        embedding_matrix = utils.save_or_load_embeds(embeds_path, vocab_path)

        self.embed = tfkl.Embedding(
            self.hparams.vocab_size,
            300,
            embeddings_initializer=tf.initializers.constant(embedding_matrix),
        )
        self.embed.trainable = self.hparams.fine_tune_embeds

        self.encoder = self.make_encoder()

        # The "non-moral" axis is actually between 0 and 1, and only 1 when the rest of
        # the components are 0.
        if self.hparams.normalize_nonmoral:
            self.encoder.add(tfkl.Dense(5, activation=tf.math.tanh))
            self.encoder.add(tfkl.Lambda(half_sphere))
        else:
            self.encoder.add(tfkl.Dense(6, activation=tf.math.tanh))

    def make_encoder(self):
        raise NotImplementedError

    def call(self, x):
        embeds = self.embed(x)
        return self.encoder(embeds)

    @tfbp.runnable
    def fit(self, data_loader):
        opt = tf.optimizers.Adam(self.hparams.learning_rate)

        # Train/validation split.
        dataset = data_loader()
        n = self.hparams.num_valid // self.hparams.batch_size
        train_dataset = dataset.skip(n).shuffle(24771 - self.hparams.num_valid)
        valid_dataset_norepeat = dataset.take(n).shuffle(self.hparams.num_valid)
        valid_dataset = iter(valid_dataset_norepeat.repeat())

        # TensorBoard writers.
        train_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "train")
        )
        valid_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "valid")
        )

        max_eval_score = float("-inf")

        while self.epoch.numpy() < self.hparams.epochs:
            for x, y in train_dataset:

                with tf.GradientTape() as tape:
                    train_loss = loss_fn(y, self(x))
                    if self.hparams.loss == "cosine_similarity":
                        train_loss = -train_loss
                grads = tape.gradient(train_loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                if self.hparams.loss == "cosine_similarity":
                    train_loss = -train_loss

                step = self.step.numpy()
                if step % 100 == 0:
                    x, y = next(valid_dataset)
                    valid_loss = loss_fn(y, self(x))
                    print(
                        "Step {} (train_loss={:.4f} valid_loss={:.4f})".format(
                            step, train_loss, valid_loss
                        ),
                        flush=True,
                    )

                    with train_writer.as_default():
                        tf.summary.scalar(
                            f"loss_{self.hparams.loss}", train_loss, step=step
                        )
                    with valid_writer.as_default():
                        tf.summary.scalar(
                            f"loss_{self.hparams.loss}", valid_loss, step=step
                        )
                        tf.summary.scalar(
                            "generalization_error",
                            tf.math.abs(train_loss - valid_loss),
                            step=step,
                        )
                else:
                    print("Step {} (train_loss={:.4f})".format(step, train_loss))

                self.step.assign_add(1)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)
            self.save()

    def _evaluate(self, dataset):
        ...

    @tfbp.runnable
    def evaluate(self, data_loader):
        self._evaluate(data_loader())
