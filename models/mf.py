"""Moral foundations classifier."""

import os.path

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
import utils


def repeat(dataset):
    while True:
        for el in dataset:
            yield el


@tfbp.default_export
class MF(tfbp.Model):
    default_hparams = {
        "rnn_layers": 1,
        "batch_size": 32,
        "vocab_size": 20000,
        "hidden_size": 512,
        "fine_tune_embeds": False,
        "loss": "huber",  # "huber" or "cos"
        "optimizer": "sgd",  # "sgd" or "adam"
        "learning_rate": 0.1,
        "num_valid": 4771,  # 24771 training examples
        "epochs": 5,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)

        embeds_path = os.path.join("data", "twitter_mf.clean.npy")
        if not os.path.isfile(embeds_path):
            word2vec = utils.load_word2vec()
            embedding_matrix = np.random.uniform(
                low=-1.0, high=1.0, size=(self.hparams.vocab_size, 300)
            )
            with open(os.path.join("data", "twitter_mf.clean.vocab")) as f:
                for i, word in enumerate(f):
                    if i >= self.hparams.vocab_size:
                        break
                    word = word.strip()
                    if word in word2vec:
                        embedding_matrix[i] = word2vec[word]
            np.save(embeds_path, embedding_matrix)
            del word2vec

        else:
            embedding_matrix = np.load(embeds_path)

        self.embed = tfkl.Embedding(
            self.hparams.vocab_size,
            300,
            embeddings_initializer=tf.initializers.constant(embedding_matrix),
        )
        self.embed.trainable = self.hparams.fine_tune_embeds

        self.encoder = tf.keras.Sequential()
        for _ in range(self.hparams.rnn_layers):
            self.encoder.add(
                tfkl.Bidirectional(
                    tfkl.GRU(self.hparams.hidden_size, return_sequences=True)
                )
            )
        self.encoder.add(tfkl.Lambda(lambda x: tf.reduce_max(x, axis=1)))
        self.encoder.add(tfkl.Dense(6, activation=tf.nn.tanh))

    def call(self, x):
        embeds = self.embed(x)
        return self.encoder(embeds)

    def train(self, dataset):
        # Loss function.
        if self.hparams.loss == "huber":
            _loss_fn = tf.losses.Huber()
        else:
            _loss_fn = tf.losses.CosineSimilarity()
        loss_fn = lambda yt, yp: tf.reduce_mean(_loss_fn(yt, yp))

        # Optimizer.
        if self.hparams.optimizer == "adam":
            opt = tf.optimizers.Adam(self.hparams.learning_rate)
        else:
            opt = tf.optimizers.SGD(self.hparams.learning_rate)

        # Train/validation split.
        n = self.hparams.num_valid // self.hparams.batch_size
        train_dataset = dataset.skip(n).shuffle(10000)
        valid_dataset = repeat(dataset.take(n).shuffle(10000))

        # TensorBoard writers.
        train_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "train")
        )
        valid_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "valid")
        )

        while self.epoch.numpy() < self.hparams.epochs:
            for x, y in train_dataset:

                with tf.GradientTape() as tape:
                    train_loss = loss_fn(y, self(x))
                grads = tape.gradient(train_loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                step = self.step.numpy()
                if step % 100 == 0:
                    x, y = next(valid_dataset)
                    valid_loss = loss_fn(y, self(x))
                    print(
                        "Step {} (train_loss={:.4f} valid_loss={:.4f})".format(
                            step, train_loss, valid_loss
                        )
                    )
                    with train_writer.as_default():
                        tf.summary.scalar("loss", train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar("loss", valid_loss, step=step)
                else:
                    print("Step {} (train_loss={:.4f})".format(step, train_loss))

                self.step.assign_add(1)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)
            self.save()

    def evaluate(self, dataset):
        valid_dataset = dataset.take(self.hparams.num_valid // self.hparams.batch_size)

        cos_sim = tf.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE)

        all_scores_pred = []
        all_scores_dumb = []
        for x, y in valid_dataset:
            scores_pred = cos_sim(y, self(x))
            scores_dumb = cos_sim(y, tf.random.normal(y.shape))
            for p, d in zip(scores_pred, scores_dumb):
                scores_pred.append(p)
                scores_dumb.append(d)

        print("baseline\t", tf.reduce_mean(all_scores_dumb).numpy())
        print("model\t", tf.reduce_mean(all_scores_pred).numpy())
        print("perfect\t", 1.0)
