#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:35:11 2017

@author: lihaoruo
"""
#============================ now A2C ============================#
import gym
import universe  # register the universe environments
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Network as NN
import ImageProcessing
from PIL import Image
import random
from rm import SumTree, Memory

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
learning_rate = 0.0001
GAMMA = 0.9
entropy_beta = 0.01
keep_prob=1.0
sess = tf.InteractiveSession()
Replay_memory = 50000
memory = Memory(capacity=Replay_memory)

actionSpace = [[], #0 For 'No Operation' action. I. e. do nothing.
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)], #1 Forward
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', True)],  #2 Forward-Nitros
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)],  #3 Forward-left
               [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'N', False)],  #4 Forwa`rd-right
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)], #5 Brake
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'N', False)],  #6 Brake-left
               [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'N', False)]]  #7 Brake-right
batch_size = 5
learning_rate = 0.000008

screenTopLeftCorner = [84,18] # Top left corner position. 84 from top, 18 from left margin.
env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()
counter = 0
saveCounter = 0
epsilon = 0.55
state = np.empty([1,102,160,2], dtype='uint8')
Q_values_est = np.empty([1,8])
action = np.random.randint(low=0, high=len(actionSpace))

action_as_1D_array = np.array(action)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state("trainrm")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
    print('-0----------------------------------------------------')
else:
    print("Could not find old network weights")
    print('---------------------------------------------------------------------------------------')
#saver.restore(sess, 'train')
while True:
    action_n = [actionSpace[:][action] for ob in observation_n]  # your agent here
    state_est = state
    observation_n, reward_n, done_n, info = env.step(action_n)
    rewardNormalized = np.array([reward_n[0]/40]) # to make the max reward closer to 1.0
    buffer_s, buffer_a, buffer_r = [], [], []

    if observation_n[0]:
        pixels_raw = observation_n[0].get("vision")[84:592, 18:818] # Size 508x800. Will be resized by 0.1
        grayscaleImg = ImageProcessing.pre_process_image(pixels_raw) # Makes a 51x80 'uint8' list
        counter += 1
        if counter%100 ==0:
            print(counter)

        if counter == 1:
            motionTracer = ImageProcessing.MotionTracer(pixels_raw) # create the object
        else:
            motionTracer.process(pixels_raw) # do stuff to create motion image

            state_raw = motionTracer.get_state() # Size 102x160x2 :
            state = np.reshape(state_raw, (102,160,2)) # 'batch' containing one entry for single estimation.
            state_4D = np.reshape(state, (1, 102, 160, 2))
            Q_values_est_raw = NN.values_est.eval(feed_dict={NN.x_state: state_4D, NN.keep_prob: 1.0})
            Q_values_est = Q_values_est_raw[0]
            buffer_v_target = []
            if counter > 399:
                if counter%10 == 0:
                    buffer_s.append(state_est)
                    buffer_a.append(action)
                    buffer_r.append(rewardNormalized)
                for r in buffer_r[::-1]:
                    Q_values_est = r + GAMMA * Q_values_est
                Q_target = Q_values_est
                
                memory.store((state_est, state, action, Q_target))

            if np.random.random() < epsilon:
                # Choose a random action.
                action = np.random.randint(low=0, high=len(actionSpace))
            else:
                #action = np.where(Q_values_est[0]==Q_values_est[0].max())[0][0] # Index of the max Q value
                a_prob = NN.a_prob.eval(feed_dict={NN.x_state: state_4D, NN.keep_prob: 1.0})
                action = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())
            action_as_1D_array = np.array([action])

            if counter == 400:
                env.reset()
                if epsilon > 0.05:
                    epsilon = epsilon*0.998
                else:
                    epsilon = 0
                learning_rate = learning_rate*0.998

                for i in range((counter-350)/25):
                    print("i:", i)
                    tree_idx, batch_memory, ISweights = memory.sample(batch_size)
                    print(np.shape(batch_memory))
                    action_batch = batch_memory[:][2]
                    print(action_batch)
                    Q_value_batch = batch_memory[:][3]
                    print(np.shape(Q_value_batch))
                    state_batch_tmp = batch_memory[:][0]
                    state_batch = state_batch_tmp#[0]
                    print(np.shape(state_batch))

                    loss = NN.loss.eval(feed_dict={NN.a: action_batch, 
                                                   NN.x_state: state_batch, 
                                                   NN.values_new: Q_value_batch, 
                                                   NN.keep_prob: 1.0})
                    #loss_is = tf.reduce_mean(ISweights * loss)
                    print(i, "loss before-----------------", loss)
                    a_loss = NN.a_loss.eval(feed_dict={NN.a: action_batch, NN.x_state: state_batch, NN.values_new: Q_value_batch, NN.keep_prob: 1.0})
                    print("a_loss", a_loss)

                    NN.optimizer.run(feed_dict={NN.a: action_batch, NN.x_state: state_batch, NN.values_new: Q_value_batch, NN.keep_prob: 1.0, NN.learning_rate: learning_rate})
                    #loss = NN.loss.eval(feed_dict={NN.a: action_batch, NN.x_state: state_batch, NN.values_new: Q_value_batch, NN.keep_prob: 1.0})
                    print(i, "loss after----------------------------------------------", loss)
                    abs_errors = NN.abs_errors.eval(feed_dict={NN.x_state: state_batch, NN.values_new: Q_value_batch, NN.keep_prob: 1.0})
                    memory.batch_update(tree_idx, abs_errors) 

                counter = 0
                saveCounter += 1
                print(saveCounter)
                imGray = Image.fromarray(grayscaleImg, mode='L')
                imGray.save('GrayScaleMedium.png')
                imMotion = Image.fromarray(state_raw[:, :, 1], mode='L')
                imMotion.save('MotionSmallMedium.png')

                if saveCounter != 0 and saveCounter%3 == 0:
                    saver.save(sess, './trainrm/duskdrive', saveCounter)

    env.render()


