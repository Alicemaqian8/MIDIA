# -*- coding: utf-8 -*-
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Imputation(object):

    def __init__(self, test_noise, n_input, weights, biases):
        self.test_noise = test_noise
        self.n_input = n_input
        self.weights = weights
        self.biases = biases
        #session
        self.sess = tf.Session()

    # Building the encoder
    def encoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),self.biases['encoder_b1']))
        return layer_1

    # Building the decoder
    def decoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        return layer_1

    def missingValImp(self):
        X = tf.placeholder("float", [None, self.n_input])
        encoder_op = self.encoder(X)
        decoder_op = self.decoder(encoder_op)
        with self.sess:
            impRes, hidden = self.sess.run([decoder_op, encoder_op], {X: self.test_noise})
            return impRes, hidden



