# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import input


class DenoiseAutoencoder(object):

    def __init__(self, train, train_noise, train_indicator, n_input, n_hidden_1, learning_rate, training_epochs,
                 batch_size):
        # network parameters
        self.train = train
        self.train_noise = train_noise
        self.train_indicator = train_indicator
        self.data = input.DataSet(train, train_noise, train_indicator)
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        # parameters
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        # weights initialize
        self.weights = self._initialize_weights()
        self.biases = self._initialize_bias()
        # session
        self.sess = tf.Session()

    def _initialize_weights(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input])),
            # 'encoder_h1': tf.Variable(tf.zeros([self.n_input, self.n_hidden_1])),
            # 'decoder_h1': tf.Variable(tf.zeros([self.n_hidden_1, self.n_input])),
        }
        return weights

    def _initialize_bias(self):
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_input])),
            # 'encoder_b1': tf.Variable(tf.zeros([self.n_hidden_1])),
            # 'decoder_b1': tf.Variable(tf.zeros([self.n_input])),
        }
        return biases

    # Building the encoder
    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        return layer_1

    # Building the decoder
    def decoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        return layer_1

    def tensorFlowPro(self):
        # tf input
        X = tf.placeholder("float", [None, self.n_input])
        X_noise = tf.placeholder("float", [None, self.n_input])
        X_indicator = tf.placeholder("float", [None, self.n_input])
        # Construct model
        encoder_op = self.encoder(X_noise)
        decoder_op = self.decoder(encoder_op)
        # Prediction
        y_pred = decoder_op * X_indicator
        # Targets (Labels) are the input data.
        y_true = X * X_indicator

        # Define loss and optimizer, minimize the squared error
        # cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        cost = tf.reduce_sum(tf.pow(y_true - y_pred, 2)) / tf.reduce_sum(X_indicator)
        # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with self.sess:
            self.sess.run(init)
            total_batch = int(self.data.num_examples / self.batch_size)
            # Training cycle
            for epoch in range(self.training_epochs):
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_xs_noise, batch_xs_indicator = self.data.next_batch(self.batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c, e, d = self.sess.run([optimizer, cost, encoder_op, decoder_op],
                                               feed_dict={X: batch_xs, X_noise: batch_xs_noise,
                                                          X_indicator: batch_xs_indicator})
                # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            # get the final parameter weights and biases
            self.weights = self.sess.run(self.weights)
            self.biases = self.sess.run(self.biases)
            print("cost: ", self.sess.run(cost, feed_dict={X: self.train, X_noise: self.train_noise,
                                                           X_indicator: self.train_indicator}))
            train_impRes = self.sess.run(y_pred, feed_dict={X: self.train, X_noise: self.train_noise,
                                                            X_indicator: self.train_indicator})
            return train_impRes

    def getWeights(self):
        return self.weights

    def getBiases(self):
        return self.biases






