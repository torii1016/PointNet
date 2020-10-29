# -*- coding:utf-8 -*-

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear, conv, batch_norm, get_dim, trans, fully_connection, conv2d, max_pool, transform, NetworkCreater, PointnetNetworkCreater

class _t_network(Layers):
    def __init__(self, name_scopes, batch_size, config):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self._config = config
        self._network_creater = NetworkCreater(config, name_scopes[0])
        self._reshape = self._config["network"]["reshape"]
        self._batch_size = batch_size


    def set_model(self, inputs, is_training=True, reuse=False):

        h  = tf.expand_dims(inputs, -1) if self._reshape else inputs
        matrix = self._network_creater.create(h, self._config, is_training, reuse)
        output_dim = self._network_creater.get_transform_output_dim()

        return  tf.reshape(matrix, [self._batch_size, output_dim, output_dim])


class _pointnet_network(Layers):
    def __init__(self, name_scopes, batch_size, pointnet_config, input_tnet_config, feature_tnet_config):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self._input_tnet = _t_network([input_tnet_config["network"]["name"]], batch_size, input_tnet_config)
        self._feature_tnet = _t_network([feature_tnet_config["network"]["name"]], batch_size, feature_tnet_config)

        self._config = pointnet_config
        self._network_creater = PointnetNetworkCreater(pointnet_config, name_scopes[0], self._input_tnet, self._feature_tnet) 

    def set_model(self, inputs, is_training=True, reuse=False):
        
        lin = self._network_creater.create(inputs, self._config, is_training, reuse)
        return lin


class PointNet(object):
    
    def __init__(self, param, pointnet_config, input_tnet_config, feature_tnet_config):

        self.batch_size = param["batch_size"]
        self.num_points = param["num_points"]
        self.lr = param["lr"]
        self.output_dim = param["output_class"]

        self.network = _pointnet_network([pointnet_config["network"]["name"]],
                                          self.batch_size,
                                          pointnet_config,
                                          input_tnet_config,
                                          feature_tnet_config)

    def set_model(self):
        
        # -- place holder ---
        self.input = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.num_points, 3])
        self.target_val = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.output_dim])

        # -- set network ---
        self.v_s = self.network.set_model(self.input, is_training=True, reuse=False)

        self.log_bce = tf.nn.sigmoid_cross_entropy_with_logits(
            logits = self.v_s,
            labels = self.target_val)
        self.cross_entropy = tf.reduce_mean(self.log_bce)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        self.v_s_wo = tf.math.sigmoid(self.network.set_model(self.input, is_training=False, reuse=True))


    def train(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _, loss = sess.run([self.train_op, self.cross_entropy], feed_dict=feed_dict)
        return _, loss

    def get_output(self, sess, input_data):
        feed_dict = {self.input: input_data}
        _ = sess.run([self.v_s_wo], feed_dict=feed_dict)
        return _