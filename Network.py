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
  #initial = tf.zeros(shape, dtype=tf.float32)
  return tf.Variable(initial)

def bias_variable(shape):
  #initial = tf.constant(0.00000000001, shape=shape)
  initial = tf.zeros(shape, dtype=tf.float32)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Model:
   
# 51 height, 80 width, 2 channel depth. Grayscale and Motion channels
x_state = tf.placeholder("float32", [None, 102, 160, 2])
# 8 possible actions were defined for this agent on 'DuskDrive' environmnet.
# 1st Convolutional layer
# Weights(shared within a filter) for first layer.
# 5x5 filter size, 2 channels, 32 different filters (size of the depth column).
# All of them initialized with positive values.
W_conv1 = weight_variable([8, 8, 2, 32])
b_conv1 = bias_variable([32])
# Only 32 because we have 32 filters. Each bias is for one filter.

h_conv1 = tf.nn.relu(conv2d(x_state, W_conv1) + b_conv1) # Size 102x160x32
h_pool1 = max_pool_2x2(h_conv1) # Finished 1st Conv. layer after ReLU and Max pooling. # Size 51x80x32

     # 2st Convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # Size 51x80x64
h_pool2 = max_pool_2x2(h_conv2) # Finished 2nd Conv. layer. Should result in 26x40x64 volume.

# 3rd Convolutional layer
W_conv3 = weight_variable([4, 4, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # Size 26x40x64
h_pool3 = max_pool_2x2(h_conv3) # Finished 2nd Conv. layer. Should result in 13x20x128 volume.

# 4th Convolutional layer
W_conv4 = weight_variable([4, 4, 128, 256])
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4) # Size 13x20x128
h_pool4 = max_pool_2x2(h_conv4) # Finished 2nd Conv. layer. Should result in 7x10x256 volume.

# 5th Convolutional layer
W_conv5 = weight_variable([4, 4, 256, 256])
b_conv5 = bias_variable([256])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5) # Size 7x10x256
h_pool5 = max_pool_2x2(h_conv5) # Finished 2nd Conv. layer. Should result in 4x5x256 volume.

# 1st Densly connected layer
W_fc1 = weight_variable([4 * 5 * 256, 400])
b_fc1 = bias_variable([400])

h_pool5_flat = tf.reshape(h_pool5, [-1, 4*5*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

# Dropout
# Prevents over-fitting. Use less than 1.0 only when training. When evaluating leave 1.0.
keep_prob = tf.placeholder("float32")

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 2nd Densly connected layer
W_fc2 = weight_variable([400, 400])
b_fc2 = bias_variable([400])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Readout layer
W_fc_advantage = weight_variable([400, 8])
b_fc_advantage = bias_variable([8])

W_fc_value = weight_variable([400, 8])
b_fc_value = bias_variable([8])

values_est = tf.matmul(h_fc2_drop, W_fc_value) + b_fc_value # Output layer.
a_prob = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc_advantage) + b_fc_advantage)
#tf.summary.scalar('Q_values', Q_values_est)




a = tf.placeholder("float", [])
learning_rate = tf.placeholder("float32", shape=[])
values_new = tf.placeholder("float32", shape=[None, 8])
    
td = values_est - values_new
log_a = tf.log(tf.clip_by_value(a_prob, 1e-20, 1.0))
entropy = -tf.reduce_sum(a_prob * log_a, reduction_indices=1)
a_loss = -tf.reduce_sum(tf.reduce_sum(tf.multiply(log_a, a), reduction_indices=1)*td + entropy*entropy_beta )

c_loss = 0.5 * tf.nn.l2_loss(values_est - values_new)
loss = a_loss + c_loss
#tf.summary.scalar('Loss', loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)


