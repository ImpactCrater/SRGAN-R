#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py


def SRGAN_g(t_image, is_train=False, reuse=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    """
    w_init = tf.initializers.variance_scaling(scale=0.01, mode='fan_avg', distribution='truncated_normal', seed=None)
    b_init = None # tf.constant_initializer(value=0.0)
    df_dim = 128
    swish = lambda x: tf.nn.swish(x)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, name='c0')
        n = GroupNormLayer(n, groups=8, act=None, name='gn0')
        temp = n

        # residual in residual dense blocks
        for i in range(8):
            n0 = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c0/%s' % i)
            n0.outputs = tf.add(n.outputs, n0.outputs)

            n1 = Conv2d(n0, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c1/%s' % i)
            n1.outputs = tf.add(n.outputs, n1.outputs)
            n1.outputs = tf.add(n0.outputs, n1.outputs)

            n2 = Conv2d(n1, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c2/%s' % i)
            n2.outputs = tf.add(n.outputs, n2.outputs)
            n2.outputs = tf.add(n0.outputs, n2.outputs)
            n2.outputs = tf.add(n1.outputs, n2.outputs)

            n3 = Conv2d(n2, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c3/%s' % i)
            n3.outputs = tf.add(n.outputs, n3.outputs)
            n3.outputs = tf.add(n0.outputs, n3.outputs)
            n3.outputs = tf.add(n1.outputs, n3.outputs)
            n3.outputs = tf.add(n2.outputs, n3.outputs)

            n4 = Conv2d(n3, df_dim, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_c4/%s' % i)
            n4.outputs = tf.add(n.outputs, n4.outputs)
            n = n4

        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_c5')
        n.outputs = tf.add(temp.outputs, n.outputs)
        # residual in residual dense blocks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.initializers.variance_scaling(scale=0.1, mode='fan_avg', distribution='truncated_normal', seed=None)
    b_init = None # tf.constant_initializer(value=0.0)
    df_dim = 64
    swish = lambda x: tf.nn.swish(x)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        n = InputLayer(input_images, name='input/images')
        n = Conv2d(n, df_dim, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c0')

        n = Conv2d(n, df_dim * 2, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c1')
        n = GroupNormLayer(n, groups=8, act=None, name='gn0')
        n = Conv2d(n, df_dim * 4, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c2')
        n = GroupNormLayer(n, groups=16, act=None, name='gn1')
        n = Conv2d(n, df_dim * 8, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c3')
        n = GroupNormLayer(n, groups=32, act=None, name='gn2')
        n = Conv2d(n, df_dim * 16, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c4')
        n = GroupNormLayer(n, groups=64, act=None, name='gn3')
        n = Conv2d(n, df_dim * 32, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c5')
        n = GroupNormLayer(n, groups=128, act=None, name='gn4')
        n = Conv2d(n, df_dim * 64, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c6')
        n = GroupNormLayer(n, groups=256, act=None, name='gn5')

        n = FlattenLayer(n, name='f0')
        n = DenseLayer(n, n_units=4096, act=swish, W_init=w_init, name='d0')
        n = DenseLayer(n, n_units=1, act=tf.identity, W_init=w_init, name='d1')
        logits = n.outputs

    return n, logits
