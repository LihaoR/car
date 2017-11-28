#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:35:11 2017

@author: lihaoruo
"""

import gym
import universe  # register the universe environments
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import Network as NN
import ImageProcessing as ip

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container

N_WORKERS = 1#multiprocessing.cpu_count()
GLOBAL_NET_SCOPE = 'global_net'
UPDATE_GLOBAL_ITER = 10
learning_rate = 0.0001
GAMMA = 0.9
entropy_beta = 0.01
sess = tf.Session()

actionSpace = [[], #0 For 'No Operation' action. I. e. do nothing.
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)], #1 Forward
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', True)],  #2 Forward-Nitros
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)],  #3 Forward-left
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'N', False)],  #4 Forward-right
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)], #5 Brake
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)],  #6 Brake-left
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'N', False)]]  #7 Brake-right

               
action = np.random.randint(low=0, high=len(actionSpace))
action_as_1D_array = np.array(action)

observation_n = env.reset()
counter = 0
total_step = 1
buffer_s, buffer_a, buffer_r = [], [], []
while True:
    #action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
    action_n = [actionSpace[:][action] for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    if observation_n[0]:
        pixels_raw = observation_n[0].get("vision")[84:592, 18:818]
        grayscaleImg = ip.pre_process_image(pixels_raw)
        counter += 1
        if counter == 1:
            motionTracer = ip.MotionTracer(pixels_raw) # create the object
        else:
            motionTracer.process(pixels_raw)
        
        state_raw = motionTracer.get_state() # Size 102x160x2 :
        state = np.reshape(state_raw, (1,102,160,2)) # 'batch' containing one entry for single estimation.
        state_4D = np.divide(state.astype(np.float32), 50)
        
        v, action_n = NN.model()
        buffer_s.append(observation_n)
        buffer_a.append(action_n)
        buffer_r.append(reward_n)
        
        v_s_, a_ = sess.run(NN.model, feed_dict={state_4D, 0.9})
        buffer_v_target = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            td = v_s_ - v
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        feed_dict_model = {NN.model.x_state: buffer_s, NN.model.keep_prob: 0.9}
        feed_dict_opt = {NN.opt.a: buffer_a, NN.opt.td: td, NN.opt.learning:0.0001, NN.opt.values_new:buffer_v_target}
        sess.run(NN.opt, feed_dict=feed_dict_opt)

    env.render()

    
    