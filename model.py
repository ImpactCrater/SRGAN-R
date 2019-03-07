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
    feature maps (n) and stride (s)
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    df_dim = 128
    swish = lambda x: tf.nn.swish(x)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, name='c0')
        n = GroupNormLayer(n, groups=8, act=None, name='gn0')
        temp = n

        # residual blocks
        for i in range(16):
            nn = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c0/%s' % i)
            nn = Conv2d(nn, df_dim, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_c1/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='res_add0/%s' % i)
            n = nn

        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_c2')
        n = ElementwiseLayer([temp, n], tf.add, name='res_add1')
        # residual blocks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    df_dim = 64
    swish = lambda x: tf.nn.swish(x)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        n = InputLayer(input_images, name='input/images')
        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, name='c0')

        n = Conv2d(n, df_dim, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c1')
        n = GroupNormLayer(n, groups=4, act=None, name='gn0')
        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c2')
        n = GroupNormLayer(n, groups=8, act=None, name='gn1')
        n = Conv2d(n, df_dim * 2, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c3')
        n = GroupNormLayer(n, groups=8, act=None, name='gn2')
        n = Conv2d(n, df_dim * 4, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c4')
        n = GroupNormLayer(n, groups=16, act=None, name='gn3')
        n = Conv2d(n, df_dim * 4, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c5')
        n = GroupNormLayer(n, groups=16, act=None, name='gn4')
        n = Conv2d(n, df_dim * 8, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c6')
        n = GroupNormLayer(n, groups=32, act=None, name='gn5')
        n = Conv2d(n, df_dim * 8, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c7')
        n = GroupNormLayer(n, groups=32, act=None, name='gn6')
        n = Conv2d(n, df_dim * 8, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c8')
        n = GroupNormLayer(n, groups=32, act=None, name='gn7')
        n = Conv2d(n, df_dim * 8, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c9')
        n = GroupNormLayer(n, groups=32, act=None, name='gn8')
        n = Conv2d(n, df_dim * 8, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c10')
        n = GroupNormLayer(n, groups=32, act=None, name='gn9')
        n = Conv2d(n, df_dim * 8, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c11')
        n = GroupNormLayer(n, groups=32, act=None, name='gn10')

        n = FlattenLayer(n, name='f0')
        n = DenseLayer(n, n_units=1024, act=swish, W_init=w_init, name='d0')
        n = DenseLayer(n, n_units=1, act=tf.identity, W_init=w_init, name='d1')
        logits = n.outputs

    return n, logits
