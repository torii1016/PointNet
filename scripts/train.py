import argparse
import sys
import numpy as np
import toml
from tqdm import tqdm
from collections import OrderedDict

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


def train(config):

    pointnet_config = toml.load(open(config["network"]["pointnet_config"]))
    input_tnet_config = toml.load(open(config["network"]["input_tnet_config"]))
    feature_tnet_config = toml.load(open(config["network"]["feature_tnet_config"]))

    batch_size = config["train"]["batch_size"]
    num_points = config["train"]["num_points"]
    epoch = config["train"]["epoch"]
    val_step = config["train"]["val_step"]
    use_gpu = config["train"]["use_gpu"]

    pointnet = PointNet(config["train"], pointnet_config, input_tnet_config, feature_tnet_config)
    pointnet.set_model()

    if use_gpu:
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

    sess = tf.compat.v1.Session(config=config)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # train
    accuracy = 0.0
    with tqdm(range(epoch)) as pbar:
        for i, ch in enumerate(pbar):
            input_datas, input_labels = data_sampler(batch_size,num_points)
            _, loss = pointnet.train(sess, input_datas, input_labels)
            pbar.set_postfix(OrderedDict(loss=loss, accuracy=accuracy))
        
            if i%val_step==0: # test
                input_datas, input_labels = data_sampler(batch_size,num_points)
                output = pointnet.get_output(sess, input_datas)[0]
                output[output>0.5]=1
                output[output<0.5]=0
                accuracy = (output==input_labels).sum().item()/batch_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/config.toml", type=str, help="default: config/config.toml")
    args = parser.parse_args()

    train(toml.load(open("config/training_param.toml")))
    #train(None)