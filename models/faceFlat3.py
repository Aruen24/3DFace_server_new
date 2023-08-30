
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops





def inference(images, keep_probability=1.0, phase_train=True,
               weight_decay=0.0, reuse=None,mode="train"):

    batch_norm_params = {
        'scale': True
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=None,
                        normalizer_params=None):
        return inception_resnet_v1_mobilenetV2(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability,
                                   reuse=reuse, mode=mode)


def inception_resnet_v1_mobilenetV2(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='mobilenet',mode="train"):

    with tf.variable_scope(scope, 'mobilenet', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):

                net = slim.conv2d(inputs, 8, 3, stride=2, activation_fn=tf.nn.relu,  padding='SAME',scope='Conv1')
                net = slim.conv2d(net, 16, 3, stride=2, activation_fn=tf.nn.relu, padding='SAME',
                                  scope='Conv2')
                net = slim.conv2d(net, 32, 3, stride=2, activation_fn=tf.nn.relu, padding='SAME',
                                  scope='Conv3')

                net = slim.conv2d(net, 32, 3, stride=2, activation_fn=tf.nn.relu, padding='SAME',
                                  scope='Conv4')
                if mode=="train":
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training)

                #net = slim.separable_conv2d(net, None, net.get_shape()[1:3], depth_multiplier=1, stride=1,
                #                            activation_fn=tf.nn.relu,
                #                            padding='VALID', scope='average')
                net = slim.separable_conv2d(net, None, net.get_shape()[1:3], depth_multiplier=1, stride=1,
                                            activation_fn=tf.nn.relu,
                                            padding='VALID', scope='average')

                net = slim.flatten(net)
                if mode == "train":
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training)

                logits = slim.fully_connected(net, 2,
                                              activation_fn=None,
                                              normalizer_fn=None,
                                              scope='logits')
                #Predictions = slim.softmax(logits)
                Predictions = tf.identity(logits, name='out')
    return logits, Predictions
