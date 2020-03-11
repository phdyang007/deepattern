import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.slim as slim

class gan(object):
    def __init__(self, input_shape=(128, 128, 1)):
        self.latent_vector = tf.placeholder(tf.float32, shape=(None, 128))
        self.generator = self.build_generator(self.latent_vector, 128*128)
        self.x_fake = self.generator
        self.x_real = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], input_shape[2]))
        # self.descriminator = self.build_discriminator(self.x)
        print('fake: ', self.x_fake)
        print('real: ', self.x_real)
        self.fake_outputs = self.build_discriminator(self.x_fake)
        self.real_outputs = self.build_discriminator(self.x_real)
        # self.g_loss, self.d_loss_real, self.d_loss_fake, self.d_loss = self.build_loss()
    def build_generator(self, input, output_dim, is_training=True, alpha=0.2):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                activation_fn=[tf.nn.relu, tf.nn.leaky_relu, tf.nn.tanh],
                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                biases_initializer=tf.constant_initializer(0.0),
                weights_regularizer=slim.l2_regularizer(0.01)):
                net = slim.fully_connected(input, 256, activation_fn=tf.nn.leaky_relu)
                net = slim.batch_norm(net)
                net = slim.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu)
                net = slim.batch_norm(net)
                net = slim.fully_connected(net, 1024, activation_fn=tf.nn.leaky_relu)
                net = slim.batch_norm(net)
                net = slim.fully_connected(net, output_dim, activation_fn=tf.nn.tanh)
        return net
    def build_discriminator(self, input, is_training=True):
        with tf.variable_scope('descriminator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=[tf.nn.relu, tf.nn.leaky_relu, tf.nn.sigmoid],
                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                biases_initializer=tf.constant_initializer(0.0),
                weights_regularizer=slim.l2_regularizer(0.01)):
                net = slim.flatten(input)
                net = slim.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu)
                net = slim.fully_connected(net, 256, activation_fn=tf.nn.leaky_relu)
                net = slim.fully_connected(net, 2, activation_fn=tf.nn.sigmoid)
        return net
    def build_loss(self):
        g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.fake_outputs.shape[0]], dtype=tf.int64),
                    logits=self.fake_outputs))
        d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.real_outputs.shape[0]], dtype=tf.int64),
                    logits=self.real_outputs))
        d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.fake_outputs.shape[0]], dtype=tf.int64),
                    logits=self.fake_outputs))
        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)
        return g_loss, d_loss_real, d_loss_fake, d_loss

if __name__ == '__main__':
    model = gan()
