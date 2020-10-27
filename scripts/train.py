import argparse
import sys
import numpy as np
import toml
from tqdm import tqdm

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

    return input_datas[data_shuffle,:,:].reshape(-1,3), input_labels[data_shuffle].reshape(-1,1)


def train(config_dict):

    batch_size = 10
    num_points = 64
    epoch = 100
    lr = 0.001


    pointnet = PointNet(3, 1, lr)
    pointnet.set_model()

    #saver = tf.compat.v1.train.Saver
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # train
    accuracy_list = []
    loss_list = []
    with tqdm(range(epoch)) as pbar:
        for i, ch in enumerate(pbar):
            input_datas, input_labels = data_sampler(batch_size,num_points)

            print("input_datas: {}".format(input_datas.shape))
            print("input_labels: {}".format(input_labels.shape))
            _, loss = pointnet.train(sess, input_datas, input_labels)
            loss_list.append(loss)
            pbar.set_postfix(OrderedDict(loss=loss))

            print(i)
    """
    # test
    accuracy = 0
    for j in range(0, test_data.shape[0], 100):
        data = test_data[j:j+100]
        label = test_label[j:j+100]
        accuracy += int(network.test(sess, data, label)[0]*data.shape[0])
    """


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/config.toml", type=str, help="default: config/config.toml")
    args = parser.parse_args()
    #train(toml.load(open(args.config)))
    train(None)