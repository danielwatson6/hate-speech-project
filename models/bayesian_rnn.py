"""Bayesian Recurrent Neural Network based language model for measuring ambiguity."""

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import utils


@tfbp.default_export
class BRNN(tfbp.Model):
    #hyperparameters from Deepmind Sonnet 
    default_hparams = {
        "batch_size": 20,
        "embed_size" : 300,
        "hidden_size" : 650,
        "num_layers" : 1,
        "epochs" : 70,
        "max_seq_len" : 35, #truncated bptt unroll length
        "lr" : 1.0 #SGD learning rate initialiser,
        "lr_decay" : 0.9 # polynomial decay power,
        "lr_decay_start" : 20,  # epoch to begin lr decay
        "fine_tune_embeds" : False,
        "vocab_path" : "",
        "vocab_size" : 20000, 
        "max_grad_norm" : 10.0,
    }
    '''
    The weights of the network, θ, are modeled as hidden random variables instead 
    of point values. THe algorithm requires a prior p(θ), and a posterior q(θ) which 
    comprises an approximation of posterior distribution over the model's weights.

    The loss function is to minimize the free energy function, obtained as a lower bound of the 
    incomplete log likelihood of the data using the variational inference scheme. 


    P(w |D ) - posterior 
    q(w |θ) - varitional posterior

    Varitional Free Energy/ Negative ELBO
    F(D, θ) = arg minθ KL[q(w|θ) || P(w)] - Eq(w|θ)[log P(D|w)]
     
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.lr = tf.Variable(self.hparams.lr, trainable=False)

        # TODO: find a way to make the model not use this.
        if not self.hparams.vocab_path:
            raise ValueError("Please specify --vocab_path=path/to/vocab.tsv")
        self.word_to_id, self.id_to_word = utils.make_word_id_maps(
            self.hparams.vocab_path, self.hparams.vocab_size,
        )   

        self.embed = tfkl.Embedding(self.hparams.vocab_size, self.hparams.embed_size)
        self.embed.trainable = False

        self.theta = tf.keras.Sequential()
        for _ in range(self.hparams.num_layers):
            self.theta.add(tfkl.LSTM(self.hparams.hidden_size, return_sequences=True))
        self.theta.add(tfkl.Dense(self.hparams.embed_size))
        self.theta.build([None, None, self.hparams.embed_size])
        
        self.d_theta = tf.reduce_sum([tf.reduce_prod(t.shape) for t in self.theta.trainable_weights])

        self.mu = tf.Variable(tf.tile([0.0], [self.d_theta]))
        self.sigma = tf.Variable(tf.tile([1.0], [self.d_theta]))

        self.opt = tf.optimizers.Adam(
            self.hparams.lr, clipnorm=self.hparams.max_grad_norm
        )

    def call(self, x):
        return self.theta(self.embed(self.word_to_id(x)))

    def loss(self, x):
        labels = self.word_to_id(x[:, 1:])
        probs = self(x[:, :-1])
        # Avoid punishing the model for "wrong" guesses on padded data.
        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        masked_loss = self.cross_entropy(labels, probs) * mask
        # Compute means by correct length, dropping any sequences of length 0.
        sequence_lengths = tf.reduce_sum(mask, axis=1)
        mean_factor = tf.map_fn(
            lambda x: tf.cond(x == 0.0, lambda: 0.0, lambda: 1.0 / x), sequence_lengths
        )
        return tf.reduce_sum(masked_loss, axis=1) * mean_factor + ... # TODO : add KL divergence term
    
    def flatten_weights(self, weights):
        return tf.concat([tf.reshape(t, [-1]) for t in weights], 0)

    def unflatten_weights(self, weights):
        shapes = [t.shape for t in self.theta.trainable_weights]
        weights = tf.split(weights, [tf.reduce_prod(shape) for shape in shapes])
        weights = [tf.reshape(t, shape) for t, shape in zip(weights, shapes)]
        return weights

    def train_step(self, x):
        # 1. Sample from the variational posterior.
        epsilon = tf.random.normal([self.d_theta])
        theta = self.unflatten_weights(self.mu + self.sigma * epsilon)
        # 2. Set neural net parameters.
        self.theta.set_weights(theta)
        # 3. Forward and backward pass.
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.theta.trainable_weights)
            loss = self.loss(x)

        g_theta = self.flatten_weights(g.gradient(loss, self.theta.trainable_weights))
        self.opt.apply_gradients([g_theta, self.mu]))
        self.opt.apply_gradients([g_theta * epsilon, self.sigma])

    @tfbp.runnable
    def fit(self, data_loader):
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

                self.train_step(batch)

                step = self.step.numpy()
                if step % 100 == 0:
                    valid_batch = next(valid_dataset_infinite)
                    valid_losses = self.loss(valid_batch)
                    valid_loss = tf.reduce_mean(valid_losses)
                    print(
                        "Step {} (train_loss={:.4f} valid_loss={:.4f})".format(
                            step, train_loss, valid_loss
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
            if self.epoch >= self.hparams.lr_decay_start:
                self.lr.assign(self.lr * self.hparams.lr_decay)
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
