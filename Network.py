#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:02:47 2017

@author: lihaoruo
"""

import tensorflow as tf
i = 1
entropy_beta = 0.01

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.zeros(shape, dtype=tf.float32)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Model:
x_state = tf.placeholder("float32", [None, 102, 160, 2])

W_conv1 = weight_variable([8, 8, 2, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_state, W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1) 

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) 
h_pool2 = max_pool_2x2(h_conv2) 

W_conv3 = weight_variable([4, 4, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) 
h_pool3 = max_pool_2x2(h_conv3) 

W_conv4 = weight_variable([4, 4, 128, 128])
b_conv4 = bias_variable([128])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4) 
h_pool4 = max_pool_2x2(h_conv4) 

W_conv5 = weight_variable([4, 4, 128, 256])
b_conv5 = bias_variable([256])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5) 
h_pool5 = max_pool_2x2(h_conv5) 

W_fc1 = weight_variable([4 * 5 * 256, 400])
b_fc1 = bias_variable([400])

h_pool5_flat = tf.reshape(h_pool5, [-1, 4*5*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float32", [])

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([400, 400])
b_fc2 = bias_variable([400])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc_advantage = weight_variable([400, 8])
b_fc_advantage = bias_variable([8])

W_fc_value = weight_variable([400, 8])
b_fc_value = bias_variable([8])

values_est = tf.matmul(h_fc2_drop, W_fc_value) + b_fc_value # Output layer.
a_prob = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc_advantage) + b_fc_advantage)

a = tf.placeholder(tf.int32, [None, ])
#a = tf.placeholder(tf.float32, [None, ])
learning_rate = tf.placeholder("float32", shape=[])
values_new = tf.placeholder("float32", shape=[None, 8])
    
td = values_est - values_new
log_a = tf.log(tf.clip_by_value(a_prob, 1e-20, 1.0))

ak = tf.one_hot(a, 8, dtype=tf.float32)
entropy = -tf.reduce_sum(a_prob * log_a, axis=1, keep_dims=True)
tmp1 = tf.multiply(log_a, ak)
tmp2 = tf.reduce_sum(tmp1, axis=1, keep_dims=True)
tmp3 = tmp2*td
tmp4 = entropy*entropy_beta
a_loss = -tf.reduce_sum(tmp3 + tmp4)

c_loss = tf.nn.l2_loss(values_est - values_new)
loss = a_loss + c_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
