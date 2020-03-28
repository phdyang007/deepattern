import os
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.contrib.slim as slim

class gan(object):
    def __init__(self, input_shape=(32, 1, 1), batch_size=128):
        self.batch_size = batch_size
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # self.latent_vector = tf.placeholder(tf.float32, shape=(None, 128))
        self.latent_vector_size = 8
        self.latent_vector = tf.random_uniform([self.batch_size, self.latent_vector_size], minval=-1.0, maxval=1.0)
        self.generator = self.build_generator(self.latent_vector, input_shape[0]*input_shape[1]*input_shape[2])
        self.x_fake = self.generator
        self.x_real = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], input_shape[2]))
        # self.descriminator = self.build_discriminator(self.x)
        print('fake: ', self.x_fake)
        print('real: ', self.x_real)
        self.fake_outputs = self.build_discriminator(self.x_fake)
        self.real_outputs = self.build_discriminator(self.x_real)
        self.g_loss, self.d_loss_real, self.d_loss_fake, self.d_loss = self.build_loss()
        self.g_opt, self.d_opt = self.build_opt(g_loss=self.g_loss, d_loss=self.d_loss)
    def build_generator(self, input, output_dim, is_training=True, alpha=0.2):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                activation_fn=[tf.nn.relu, tf.nn.leaky_relu, tf.nn.tanh],
                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                biases_initializer=tf.constant_initializer(0.0),
                weights_regularizer=slim.l2_regularizer(0.01)):
                net = slim.fully_connected(input, 16, activation_fn=tf.nn.leaky_relu)
                net = slim.batch_norm(net)
                net = slim.fully_connected(net, 32, activation_fn=tf.nn.leaky_relu)
                net = slim.batch_norm(net)
                net = slim.fully_connected(net, 64, activation_fn=tf.nn.leaky_relu)
                net = slim.batch_norm(net)
                net = slim.fully_connected(net, output_dim, activation_fn=tf.nn.tanh)
        return net
    def build_discriminator(self, input, is_training=True):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                activation_fn=[tf.nn.relu, tf.nn.leaky_relu, tf.nn.sigmoid],
                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                biases_initializer=tf.constant_initializer(0.0),
                weights_regularizer=slim.l2_regularizer(0.01)):
                net = slim.flatten(input)
                net = slim.fully_connected(net, 32, activation_fn=tf.nn.leaky_relu)
                net = slim.fully_connected(net, 16, activation_fn=tf.nn.leaky_relu)
                net = slim.fully_connected(net, 2, activation_fn=tf.nn.sigmoid)
        return net
    def build_loss(self):
        ones = tf.ones([self.batch_size], dtype=tf.int64)
        zeros = tf.zeros([self.batch_size], dtype=tf.int64)
        g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(ones, 2),
                    logits=self.fake_outputs))
        one_label = tf.one_hot(ones, 2)
        one_label = one_label * 0.8
        zero_label = tf.one_hot(zeros, 2)
        zero_label = zero_label * 0.2
        one_label_one_hot = one_label + zero_label
        one_label = tf.one_hot(ones, 2)
        zero_label = tf.one_hot(zeros, 2)
        one_label = one_label * 0.2
        zero_label = zero_label * 0.8
        zero_label_one_hot = one_label + zero_label
        d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=one_label_one_hot,
                    logits=self.real_outputs))
        d_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=zero_label_one_hot,
                    logits=self.fake_outputs))
        """
        g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=self.fake_outputs))
        d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=self.real_outputs))
        d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=self.fake_outputs))
        """
        d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)
        return g_loss, d_loss_real, d_loss_fake, d_loss
    def build_opt(self, g_loss, d_loss):
        var_list = [v for v in tf.trainable_variables()]
        g_var = [v for v in var_list if 'generator' in v.name]
        d_var = [v for v in var_list if 'discriminator' in v.name]
        print('g_var:', g_var)
        print('d_var:', d_var)
        g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(g_loss, var_list=g_var)
        d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(d_loss, var_list=d_var)
        return g_opt, d_opt

def train(model, data, conf):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=100)
        lr = conf.learning_rate
        for step in range(conf.max_iter):
            train_data = data.getTrainBatch(conf.batch_size)
            _ = sess.run([model.g_opt], feed_dict={model.x_real: train_data, model.learning_rate: lr})
            g_loss, d_loss, d_loss_real, d_loss_fake,  _, _ = sess.run([model.g_loss, model.d_loss, model.d_loss_real, model.d_loss_fake, model.g_opt, model.d_opt], feed_dict={model.x_real: train_data, model.learning_rate: lr})
            if step % 1000 == 0:
                print('Step[%d/%d]: g_loss=%.4f, d_loss=%.4f, d_loss_real: %.4f, d_loss_fake: %.4f'%(step, conf.max_iter, g_loss, d_loss, d_loss_real, d_loss_fake))
            if (step % conf.decay_step == conf.decay_step - 1):
                lr *= conf.decay_rate

            if (step % conf.save_step == conf.save_step - 1):
                saver.save(sess, os.path.join(conf.model_dir, 'model-'+str(step)+'.ckpt'))

def test(model, conf):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=100)
        checkpoint_file = tf.train.latest_checkpoint(conf.model_dir)
        print(checkpoint_file)
        print(conf.model_dir)
        saver.restore(sess, checkpoint_file)
        generated_fake = None
        for step in range(conf.test_iter):
            x_fake = sess.run(model.x_fake)
            if (generated_fake is None):
                generated_fake = x_fake
            else:
                generated_fake = np.concatenate((generated_fake, x_fake), axis=0)
        print(generated_fake.shape)
        generated_fake = generated_fake.astype(np.float32)
        fake_list = [v.tolist() for v in generated_fake]
        # fake_list = [v[0] for v in fake_list]
        df = pd.DataFrame()
        df['vector'] = fake_list
        print(df.shape)
        # df.to_msgpack(os.path.join(conf.test_save_dir, 'vector.msgpack'))
        df.to_pickle(os.path.join(conf.test_save_dir, 'vector.pkl'))
            
            
        
class conf(object):
    batch_size = 128
    max_iter = 1000000
    learning_rate = 0.01
    decay_step = 10000
    decay_rate = 0.95
    model_dir = '../models/tc1/gan/'
    data_dir = '../models/tc1/'
    test_save_dir = '../models/tc1/gan/test/'
    save_step = 10000
    test_iter = 1000
    def __init__(self, flag):
        self.data_dir = os.path.join('models', flag)
        self.model_dir = os.path.join(os.path.join('models', flag), 'gan')
        self.test_save_dir = os.path.join(self.model_dir, 'test')


class Data(object):
    def __init__(self, filedir):
        self.filename = os.path.join(filedir, 'test/noise_data.msgpack')
        self.filedir = filedir
        self.data = pd.read_msgpack(self.filename)
        self.data = self.data[self.data['valid'] == 1]
        self.data = self.data['vector'].tolist()
        self.data = np.array(self.data)
        self.current_ptr = 0
        print(self.data.shape)
        

    def getTrainBatch(self, batch_size):
        if (self.current_ptr + batch_size >= self.data.shape[0]):
            self.current_ptr = 0
        if (self.current_ptr == 0):
            np.random.shuffle(self.data)
        next_ptr = self.current_ptr + batch_size
        batch = self.data[self.current_ptr:next_ptr]
        self.current_ptr = next_ptr
        return np.reshape(batch, (batch_size, 32, 1, 1))

    @staticmethod
    def merge(filedir):
        df1 = pd.DataFrame()
        for i in range(100):
            try:
                filename = os.path.join(filedir, 'test/vector_'+str(i)+'.msgpack')
                print(filename)
                df = pd.read_msgpack(filename)
                print(df.shape)
                print(df)
                df1.append(df)
            except Exception as e:
                print(e)
        df1.to_msgpack(os.path.join(filedir, 'test/vector.msgpack'))
    

    
if __name__ == '__main__':
    if sys.argv[1] == 'train':
        c = conf(sys.argv[2])
        model = gan()
        data = Data(c.data_dir)
        train(model, data, c)
    elif sys.argv[1] == 'test':
        c = conf(sys.argv[2])
        model = gan(batch_size=100)
        test(model, c)
    else:
        print('read')
        # df = pd.read_msgpack('../models/tc1/gan/test/vector.msgpack')
        df = pd.read_pickle('../models/tc1/gan/test/vector.pkl')
        print(df)
        print(df.shape)
