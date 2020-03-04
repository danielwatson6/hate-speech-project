import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp

import glob
import matplotlib.pyplot as plt
import PIL
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

        self.encoder = tfkl.Sequential()
        self.decoder = tfkl.Sequential()
        for size in self.hparams.hidden_sizes:
            self.encoder.add(tfkl.Dense(size, activation=tf.nn.relu))
            self.decoder.add(tfkl.Dense(size, activation=tf.nn.relu))

        self.encoder.add(tfkl.Dense(self.hparams.latent_size * 2, activation=None))
        self.decoder.add(tfkl.Dense(28 * 28, activation=None))

        # tensorflow tutorial has this input layer...tf.keras.layers.InputLayer(input_shape=(latent_dim,))

        # for size in self.hparams.hidden_sizes:
        #     self.decoder.add(tfkl.Dense(size))
        # self.decoder.add(tfkl.Dense(28 * 28, activation=tf.nn.sigmoid))

    def sample(self, s=None):  # samples from a Gaussian distribution
        if s is None:
            s = tf.random.normal(shape=(100, self.hparams.latent_size))
        return self.decode(s, sigmoid=True)

    def encode(self, x):
        mean, var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, var

    def decode(self, z, sigmoid=False):
        logits = self.decoder(z)
        if sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    # samples from Gaussian, multiplies by standard deviation and adds mean
    def reparameterize(self, mean, var):
        s = tf.random.normal(shape=mean.shape)
        return s * tf.exp(var * 0.5) + mean

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def generate_and_save_images(self, epoch, test_input):
        predictions = self.sample(test_input)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap="gray")
            plt.axis("off")

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
        plt.show()

    @tfbp.runnable
    def fit(self, data_loader):
        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        random_vector_for_generation = tf.random.normal(
            shape=[self.hparams.num_examples_to_generate, self.hparamslatent_dim]
        )

        self.generate_and_save_images(0, random_vector_for_generation)

        train_data, valid_data = data_loader()
        valid_dataset_infinite = utils.infinite(valid_data)
        optimizer = tf.keras.optimizers.Adam(self.hparams.learning_rate)

        # Tensorboard writers
        train_writer = self.make_summary_writer("train")
        valid_writer = self.make_summary_writer("valid")

        while self.epoch.numpy() < self.hparams.epochs:
            for batch in train_data:
                self.compute_apply_gradients(batch, optimizer)
                train_loss = tf.reduce_mean(self.compute_loss(batch))
                elbo = -train_loss.result()
                display.clear_output(wait=False)
                step = self.step.numpy()
                if step % 100 == 0:
                    valid_batch = next(valid_dataset_infinite)
                    valid_loss = tf.reduce_mean(self.compute_loss(valid_batch))
                    print(
                        "Step {} (train_loss={:.4f} valid_loss={:.4f} elbo = {:.4f})".format(
                            step, train_loss, valid_loss, elbo
                        ),
                        flush=True,
                    )
                    with train_writer.as_default():
                        tf.summary.scalar("loss", train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar("loss", valid_loss, step=step)
            self.generate_and_save_images(
                self.epoch.numpy(), random_vector_for_generation
            )
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

