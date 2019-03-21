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
            n0_0 = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c0/%s' % i)
            n0_1 = ConcatLayer([n, n0_0], concat_dim=-1, name='res_concat0/%s' % i)

            n1_0 = Conv2d(n0_1, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c1/%s' % i)
            n1_1 = ConcatLayer([n, n0_0, n1_0],concat_dim=-1, name='res_concat1/%s' % i)

            n2_0 = Conv2d(n1_1, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c2/%s' % i)
            n2_1 = ConcatLayer([n, n0_0, n1_0, n2_0], concat_dim=-1, name='res_concat2/%s' % i)

            n3_0 = Conv2d(n2_1, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c3/%s' % i)
            n3_1 = ConcatLayer([n, n0_0, n1_0, n2_0, n3_0], concat_dim=-1, name='res_concat3/%s' % i)

            n4_0 = Conv2d(n3_1, df_dim, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_c4/%s' % i)
            n4_1 = ElementwiseLayer([n, n4_0], tf.add, name='res_add0/%s' % i)
            n = n4_1

        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res_c5')
        n = ElementwiseLayer([temp, n], tf.add, name='res_add1')
        # residual in residual dense blocks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.initializers.variance_scaling(scale=0.01, mode='fan_avg', distribution='truncated_normal', seed=None)
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
