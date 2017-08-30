# -*- coding:utf-8 -*-
from utils import (
  read_data,
  input_setup,
  imsave,
  merge
)
import scipy.io
import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class SRCNN(object):

  def __init__(self,
               sess,
               n_inputs=128,
               n_outputs=128,
               batch_size=128,
               n_steps=128,
               out_step=4,
               n_hidden_units=128,
               is_train=True,
               checkpoint_dir=None,
               sample_dir=None):

    self.sess = sess
    self.n_inputs = n_inputs
    self.n_outputs = n_outputs
    self.n_steps = n_steps
    self.out_step = out_step
    self.batch_size = batch_size
    self.n_hidden_units = n_hidden_units
    self.is_train = is_train
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()
  def RNN(self,X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, self.n_inputs])#-1 表示自动补齐，使得和原来array中的总数一臿    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    #forget_bias=1.0 的意思就是先不要打开forget的gate即不忘记之前的state
    #n_hidden_unit 是lstm cell中的神经元有多少丿
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        def lstm_cell():
            if self.is_train:
                lstm = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
                lstm = tf.contrib.rnn.DropoutWrapper(lstm)
            else:
                lstm = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
            return lstm
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.out_step)],state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, m_state)
    init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    #实际上，上面这个函数里面也有递归，每一次time step都要在一个cell里面递归一次，28个n_step结束后才算一个batch结束〿    #一共有batch_size个cell，这个cell结束后state传给下一个cell。这些batch_size个cell可以看成一个整使    # http://r2rt.com/styles-of-truncated-backpropagation.html
    # hidden layer for output as the final results
    #############################################
    #final = final_state[0][1]
    final = tf.concat([final_state[i][1] for i in range(self.out_step)],0)
    #final=tf.cast(final,[-1,self.n_hidden_units])
    results = tf.matmul(final, weights['out']) + biases['out']
    results = tf.maximum(0.0,results)
    # # or
    # unpack to list [(batch, outputs)..] * steps
    #if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        #outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    #else:
        #outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    #results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
    return results

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs]) #行数为n_input,列数为n_step,none 代表 n_sample
    self.labels = tf.placeholder(tf.float32, [None, self.out_step,self.n_outputs])
    # Define weights
    weights = {
        # (28, 128) input＿output
        # rnn cell的input和output各有一个hidden layer 夹着宿
        'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])), #生成一个（n_inputs,n_hidden_units）的矩阵 n_inputs 为row number
        																 #output丿28 means 和weights矩阵相乘后会生成一个column number丿28的矩阿        # (128, 10)
        'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_outputs]))
    }

    biases = {
        # (128, )
        'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
        # (10, )
        'out': tf.Variable(tf.constant(0.1, shape=[self.n_outputs, ]))
    }

    '''
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')

    self.weights = {
      'w1': tf.Variable(tf.random_normal([21, 21, 1, 48], stddev=1e-3), name='w1'),
      'w2': tf.Variable(tf.random_normal([1, 1, 48, 24], stddev=1e-3), name='w2'),
      'w3': tf.Variable(tf.random_normal([9, 9, 24, 1], stddev=1e-3), name='w3')
    }
    self.biases = {
      'b1': tf.Variable(tf.zeros([48]), name='b1'),
      'b2': tf.Variable(tf.zeros([24]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }
    '''

    self.pred = self.RNN(self.images, weights, biases)
    #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels))
    labels = tf.reshape(self.labels,[-1,self.n_outputs])
    self.loss = tf.reduce_mean(tf.square(labels - self.pred))
    #correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    '''
    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    '''
    self.saver = tf.train.Saver(tf.all_variables(),max_to_keep = 10)

  def train(self, config):
    if config.is_train:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
      if not os.path.isfile(data_dir):
        input_setup(self.sess, config)
    else:
      input_setup(self.sess, config)
    if config.is_train:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    train_data, train_label = read_data(data_dir)
    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

    tf.global_variables_initializer().run()

    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      with open("./log",'a') as f:
        for ep in range(config.epoch):
          # Run by batch images
          batch_idxs = len(train_data) // config.batch_size
          err_total=0.0
          for idx in range(0, batch_idxs):
            batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
            batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
            counter += 1
            _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
            '''
            outputs,final_state = self.sess.run([self.outputs, self.final_state], feed_dict={self.images: batch_images, self.labels: batch_labels})
            print outputs.shape
            '''
            if counter % 10 == 0:
              print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                % ((ep+1), counter, time.time()-start_time, err))

            if counter % 500 == 0:
              self.save(config.checkpoint_dir, counter)
            err_total+=err
          print("Epoch: [%2d], loss: [%.8f]" \
                % ((ep+1), err_total))
          f.write("Epoch: [%2d], loss: [%.8f]\n" \
                % ((ep+1), err_total))
          f.flush()
    else:
      print("Testing...")

      #eval():将字符串str当成有效的表达式来求值并返回计算结果
      result = self.pred.eval({self.images: train_data})
      #result = self.sess.run(self.pred,{self.images:train_data})
      print(type(result))
      print(result.shape)
      print(type(train_data))
      print(train_data.shape)
      result1=np.vstack((np.squeeze(train_data),result))
      plt.imshow(result1)
      plt.pause(10)
      #result = merge(result, [nx, ny])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path_test = os.path.join(image_path, "test_image.mat")
      image_path_label= os.path.join(image_path, 'label_image.mat')
      scipy.io.savemat(image_path_test,{'R':result})
      scipy.io.savemat(image_path_label,{'R':train_label})
      #imsave(result[0], image_path)

  def model(self):
    conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
    conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.n_steps)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.n_steps)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
