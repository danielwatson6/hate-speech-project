"""Moral foundations classifier."""

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
import utils


@tfbp.default_export
class MF(tfbp.Model):
    default_hparams = {
        "batch_size": 32,
        "vocab_size": 28371,  # all vocabulary
        "fine_tune_embeds": False,
        "loss": "cos",  # "huber" or "cos"
        "optimizer": "sgd",  # "sgd" or "adam"
        "learning_rate": 0.1,
        "num_valid": 4771,  # 24771 training examples
        "epochs": 20,
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

        self.encoder = self.make_encoder()
        self.encoder.add(tfkl.Dense(6, activation=tf.math.tanh))

    def make_encoder(self):
        raise NotImplementedError

    def call(self, x):
        embeds = self.embed(x)
        return self.encoder(embeds)

    def fit(self, data_loader):
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
        dataset = data_loader()
        n = self.hparams.num_valid // self.hparams.batch_size
        train_dataset = dataset.skip(n).shuffle(10000)
        valid_dataset = iter(dataset.take(n).shuffle(10000).repeat())

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
                    if self.hparams.loss == "cos":
                        train_loss = -train_loss
                grads = tape.gradient(train_loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                if self.hparams.loss == "cos":
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
                        tf.summary.scalar(self.hparams.loss, train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar(self.hparams.loss, valid_loss, step=step)
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
            cos_score, mae_scores = self._evaluate(dataset)
            with valid_writer.as_default():
                tf.summary.scalar("total_cos", cos_score, step=step)
                tf.summary.scalar("avg_mae", tf.reduce_mean(mae_scores), step=step)
                tf.summary.histogram("mae", mae_scores, step=step)

            if cos_score > max_eval_score:
                self.save()

    def _evaluate(self, dataset):
        valid_dataset = dataset.take(self.hparams.num_valid // self.hparams.batch_size)
        cos_sim = tf.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE)
        mae = lambda x, y: tf.math.abs(x - y)
        all_cos_sim = []
        all_mae = []
        for x, y in valid_dataset:
            y_pred = self(x)
            for a, b in zip(cos_sim(y, y_pred), mae(y, y_pred)):
                all_cos_sim.append(a)
                all_mae.append(b)

        cos_score = tf.reduce_mean(all_cos_sim).numpy()
        mae_scores = tf.reduce_mean(all_mae, axis=0).numpy()
        print("cos\t", cos_score)
        print("authority\t", mae_scores[0])
        print("fairness\t", mae_scores[1])
        print("care\t", mae_scores[2])
        print("loyalty\t", mae_scores[3])
        print("purity\t", mae_scores[4])
        print("non_moral\t", mae_scores[5], flush=True)
        return cos_score, mae_scores

    def evaluate(self, data_loader):
        dataset = data_loader()
        self._evaluate(dataset)
