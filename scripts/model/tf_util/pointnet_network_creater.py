# -*- coding:utf-8 -*-

import sys
import tensorflow as tf

from .network_creater import NetworkCreater

class PointnetNetworkCreater(NetworkCreater):
    def __init__(self, config, name_scope, input_tnet, feature_tnet):
        super().__init__(config, name_scope)
        self._creater = {"conv2d":self._conv2d_creater,
                        "fc":self._fc_creater,
                        "reshape":self._reshape_creater,
                        "transform":self._transform_creater,
                        "maxpool":self._maxpool_creater,
                        "tnet":self._tnet_creater}
        self._input_tnet = input_tnet
        self._feature_tnet = feature_tnet


    def _tnet_creater(self, inputs, data, is_training=None, reuse=None):

        if data["format"]=="point":
            transform_matrix = self._input_tnet.set_model(inputs, is_training=is_training, reuse=reuse)
            h = tf.matmul(inputs, transform_matrix)
            h = tf.expand_dims(h, -1)

        elif data["format"]=="feature":
            transform_matrix = self._feature_tnet.set_model(inputs, is_training=is_training, reuse=reuse)
            h = tf.matmul(tf.squeeze(inputs, axis=[2]), transform_matrix)
            h = tf.expand_dims(h, [2])
        
        else:
            print("##### format error, check toml file ('format') #####")
            sys.exit()

        return h