from model import SRCNN
from utils import input_setup

import numpy as np
import tensorflow as tf

import pprint
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
flags = tf.app.flags
flags.DEFINE_integer("epoch", 2000, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("n_inputs", 128, "The size of image to use [33]")
flags.DEFINE_integer("n_outputs", 128, "The size of image to use [33]")
flags.DEFINE_integer("n_steps", 128, "The size of label to produce [21]")
flags.DEFINE_integer("n_hidden_units", 128, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  with tf.Session() as sess:
    srcnn = SRCNN(sess, 
                  n_inputs=FLAGS.n_inputs, 
                  n_outputs=FLAGS.n_outputs, 
                  batch_size=FLAGS.batch_size,
                  n_steps=FLAGS.n_steps, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
