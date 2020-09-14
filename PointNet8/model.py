import tensorflow as tf
import math
import time
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32,
                               shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """  # No colors
    num_classes = 8  # was 13
    conv1_size = 64 / 4
    conv2_size = 64 / 4
    conv3_size = 64 / 4
    conv4_size = 128 / 4
    conv5_size = 1024 / 4
    fc1_size = 256 / 4
    fc2_size = 128 / 4
    conv6_size = 512 / 4
    conv7_size = 256 / 4

    batch_size = point_cloud.get_shape()[0].value  # extracts the batch_size from pointclouds_pl
    num_point = point_cloud.get_shape()[1].value  # extracts the batch_size from pointclouds_pl

    input_image = tf.expand_dims(point_cloud, -1)  # builds an image of point cloud: size 24 x 4096 x 9 x 1
    # CONV                                         # extra - 1 for the gray image, -> empty tensor
    net = tf_util.conv2d(input_image, conv1_size, [1, 9], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)  # hard coded 9!
    net = tf_util.conv2d(net, conv2_size, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, conv3_size, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, conv4_size, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, conv5_size, [1, 1], padding='VALID', stride=[1, 1],
                                  bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # Important:
    # Only use the dimensions of the point cloud but not the values (the received point_cloud is empty tensor)

    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point, 1], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, fc1_size, bn=True, is_training=is_training, scope='fc1',
                                       bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, fc2_size, bn=True, is_training=is_training, scope='fc2',
                                       bn_decay=bn_decay)
    print(pc_feat1)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

    # CONV
    net = tf_util.conv2d(points_feat1_concat, conv6_size, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv6')
    net = tf_util.conv2d(net, conv7_size, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv7')
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')  # drop_up - does need to change
    net = tf_util.conv2d(net, num_classes, [1, 1], padding='VALID', stride=[1, 1],  # hard coded 13!
                         activation_fn=None, scope='conv8')
    net = tf.squeeze(net, [2])

    return net


def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)


if __name__ == "__main__":  # Hard coded values!!
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32, 4096, 9))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a: np.random.rand(32, 4096, 9)})
            print(time.time() - start)
