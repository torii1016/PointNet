import argparse
import sys
import os
import numpy as np
import toml
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime

import tensorflow as tf

from model.pointnet import PointNet

def data_sampler(batch_size, num_points):
    half_batch_size = int(batch_size/2)
    normal_sampled = np.random.normal(0, 1, (half_batch_size, num_points, 3))
    uniform_sampled = np.random.rand(half_batch_size, num_points, 3)
    normal_labels = np.ones(half_batch_size)
    uniform_labels = np.zeros(half_batch_size)

    input_datas = np.concatenate([normal_sampled, uniform_sampled], axis=0)
    input_labels = np.concatenate([normal_labels, uniform_labels], axis=0)

    data_shuffle = np.random.permutation(list(range(batch_size)))

    return input_datas[data_shuffle,:,:], input_labels[data_shuffle].reshape(-1,1)


class Trainer(object):
    def __init__(self, config):
        pointnet_config = toml.load(open(config["network"]["pointnet_config"]))
        input_tnet_config = toml.load(open(config["network"]["input_tnet_config"]))
        feature_tnet_config = toml.load(open(config["network"]["feature_tnet_config"]))

        self._batch_size = config["train"]["batch_size"]
        self._num_points = config["train"]["num_points"]
        self._epoch = config["train"]["epoch"]
        self._val_step = config["train"]["val_step"]
        self._use_gpu = config["train"]["use_gpu"]
        self._save_model_path = config["train"]["save_model_path"]
        self._save_model_name = config["train"]["save_model_name"]

        self._pointnet = PointNet(config["train"], pointnet_config, input_tnet_config, feature_tnet_config)
        self._pointnet.set_model()

        if self._use_gpu:
            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.8,
                    allow_growth=True
                )
            )
        else:
            config = tf.ConfigProto(
                device_count = {'GPU': 0}
            )

        self._sess = tf.compat.v1.Session(config=config)
        init = tf.compat.v1.global_variables_initializer()
        self._sess.run(init)
        self._saver = tf.train.Saver()

        self._accuracy = 0.0

        self._tensorboard_path = "./logs/" + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


    def _save_model(self):
        os.makedirs(self._save_model_path, exist_ok=True)
        self._saver.save(self._sess, self._save_model_path+"/"+self._save_model_name)


    def _save_tensorboard(self, loss):
        with tf.name_scope('log'):
            tf.summary.scalar('loss', loss)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self._tensorboard_path, self._sess.graph)


    def train(self):
        with tqdm(range(self._epoch)) as pbar:
            for i, ch in enumerate(pbar): #train
                input_datas, input_labels = data_sampler(self._batch_size, self._num_points)
                _, loss = self._pointnet.train(self._sess, input_datas, input_labels)
                pbar.set_postfix(OrderedDict(loss=loss, accuracy=self._accuracy))

                self._save_tensorboard(loss)

                if i%self._val_step==0: #test
                    input_datas, input_labels = data_sampler(self._batch_size, self._num_points)
                    output = self._pointnet.get_output(self._sess, input_datas)[0]
                    output[output>0.5]=1
                    output[output<0.5]=0
                    self._accuracy = (output==input_labels).sum().item()/self._batch_size

                    self._save_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/config.toml", type=str, help="default: config/config.toml")
    args = parser.parse_args()

    trainer = Trainer(toml.load(open("config/training_param.toml")))
    trainer.train()