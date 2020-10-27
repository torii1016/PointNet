# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim

class _t_network(Layers):
    def __init__(self, name_scopes, layer_channels_first, layer_channels_latter, ksize):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.name_scope = name_scopes[0]
        self.layer_channels_first = layer_channels_first #[64,128,1024]
        self.layer_channels_latter = layer_channels_latter #[512,256,9]
        self.ksize = ksize

    def set_model(self, inputs, is_training = True, reuse = False):
        h  = inputs
        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            for i, s in enumerate(self.layer_channels_first):
                lin = linear(i, h, s)
                h = tf.nn.relu(lin)
                h = batch_norm(i, h, is_training)
            
        #h = tf.nn.max_pool1d(tf.reshape(h, [-1, 1024, 16]), self.ksize, strides=None, padding="VALID") 
        h = tf.nn.max_pool1d(tf.reshape(h, [-1,1024,64]), [1,1,64], strides=None, padding="VALID", name=self.name_scope) 
        h = tf.reshape(h, [-1, 1024])

        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            for i, s in enumerate(self.layer_channels_latter):
                lin = linear(i+len(self.layer_channels_first), h, s)
                h = tf.nn.relu(lin)
                h = batch_norm(i+len(self.layer_channels_first), h, is_training)
            matrix = lin

        return tf.matmul(inputs, matrix)


class _pointnet_network(Layers):
    def __init__(self, name_scopes, ksize, output_dim):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.name_scope = name_scopes[0]
        self.ksize = ksize
        self.input_t_net = _t_network(["InputTNet"], [64, 128, 1024], [512, 256, 9], 64)
        self.feature_t_net = _t_network(["FeatureTNet"], [64, 128, 1024], [512, 256, 4096], 64)
        self.layer_channels_first = [64, 64]
        self.layer_channels_second = [64, 128, 1024]
        self.layer_channels_third = [512, 256, output_dim]

    def set_model(self, inputs, is_training = True, reuse = False):

        h = self.input_t_net.set_model(inputs, is_training=is_training, reuse=reuse)

        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            for i, s in enumerate(self.layer_channels_first):
                lin = linear(i, h, s)
                h = tf.nn.relu(lin)
                h = batch_norm(i, h, is_training)

        h = self.feature_t_net.set_model(h, is_training=is_training, reuse=reuse)

        with tf.compat.v1.variable_scope(self.name_scope, reuse = reuse):
            for i, s in enumerate(self.layer_channels_second):
                lin = linear(i+len(self.layer_channels_first), h, s)
                h = tf.nn.relu(lin)
                h = batch_norm(i+len(self.layer_channels_first), h, is_training)

            #h = tf.nn.max_pool1d(h, self.ksize, strides=None, padding="VALID") 
            h = tf.nn.max_pool1d(tf.reshape(h, [-1, 1024, 64]), [1,1,64], strides=None, padding="VALID") 
            h = tf.reshape(h, [-1, 1024])

            for i, s in enumerate(self.layer_channels_third):
                lin = linear(i+len(self.layer_channels_first)+len(self.layer_channels_second), h, s)
                lin = tf.nn.relu(lin)
                lin = batch_norm(i+len(self.layer_channels_first)+len(self.layer_channels_second), lin, is_training)
                h = tf.nn.dropout(h, 0.3)
        return lin


class PointNet(object):
    
    def __init__(self, input_dim, output_dim, lr):
        self.network = _pointnet_network(["PointNet"], 64, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.point_num = 64
        self.lr = lr

    def set_model(self):
        
        # -- place holder ---
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.target_val = tf.placeholder(tf.float32, [None, self.output_dim])

        # -- set network ---
        self.v_s = self.network.set_model(self.input, is_training = True, reuse = False)

        self.log_soft_max = tf.nn.softmax_cross_entropy_with_logits(
            logits = self.v_s,
            labels = self.target_val)
        self.cross_entropy = tf.reduce_mean(self.log_soft_max)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        # -- for test --
        self.v_s_wo = self.network.set_model(self.input, is_training = False, reuse = True)
        self.correct_prediction = tf.equal(tf.argmax(self.v_s_wo,1), tf.argmax(self.target_val,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.output_probability = tf.nn.softmax(self.v_s_wo)


    def train(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _, loss = sess.run([self.train_op, self.cross_entropy], feed_dict = feed_dict)
        return _, loss

    def test(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _ = sess.run([self.accuracy], feed_dict = feed_dict)
        return _

    def get_output(self, sess, input_data):
        feed_dict = {self.input: input_data}
        _ = sess.run([self.output_probability], feed_dict = feed_dict)
        return _