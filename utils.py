"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""
import copy
import scipy.io
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file

  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  """
  Preprocess single image file
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = scipy.io.loadmat(path)
  label_ = image

  #input_ =
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset

    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.mat"))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.mat"))

  return data

def make_data(sess, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def mixSegment(data,label):
  traindata=[]
  testdata =[]
  numbers=len(data)
  mixSeq=list(range(numbers))
  random.shuffle(mixSeq)
  traindata=data[mixSeq]
  testdata=label[mixSeq]
  return traindata,testdata

#for train:read image files and make them to h5 file format.
#for test: return nx,ny.
def input_setup(sess, config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  Stride=1
  MAX_S=40
  Pad=10
  # Load data path
  if config.is_train:
    dataset="train_data"
    data_dir=os.path.join(os.getcwd(),dataset)
    data = glob.glob(os.path.join(data_dir,"*.mat"))
  else:
    dataset='test_data'
    data_dir=os.path.join(os.getcwd(),dataset)
    data = glob.glob(os.path.join(data_dir,"phan.mat"))

  sub_input_sequence = []
  sub_label_sequence = []

  if config.is_train:
    for i in range(len(data)):
      '''
       input_, label_ = preprocess(data[i], config.scale)

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
        '''
      imgg=scipy.io.loadmat(data[i])
      image=imgg['R']
      image=image/image.max()
      h,w=image.shape
      for y in range(config.n_steps, (h-(config.out_step))):
        sub_input = image[(y-config.n_steps):y,:]
        sub_label = (image[y:(y+config.out_step),:])
        '''
        image_path=os.path.join(os.getcwd(),config.sample_dir)
        image_name='train_image'+str(y*10+x)+'.mat'
        image_path=os.path.join(image_path,image_name)
        scipy.io.savemat(image_path,{'R':RRR})
        '''
        # Make channel value
        #sub_input = sub_input.reshape([config.n_steps, config.n_inputs, 1])
        #sub_label = sub_label.reshape([config.out_step, config.n_outputs, 1])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  else:
    imgg=scipy.io.loadmat(data[0])
    image=imgg['R']
    image = image/image.max()
    h,w=image.shape
    sub_input=image[:config.n_steps,:]
    sub_label=image[config.n_steps:(config.n_steps+config.out_step),:]

    #sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
    #sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
    sub_input_sequence.append(sub_input)
    sub_label_sequence.append(sub_label)
    '''
    input_, label_ = preprocess(data[2], config.scale)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape
    '''
    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    #nx = ny = 0
    '''
    for x in range(0, MAX_S , Stride):
      nx += 1; ny = 0
      for y in range(1, (h-x) , Pad):
        ny += 1
        sub_input = copy.deepcopy(image)
        sub_input[y:(y+x)]=0
        sub_label = copy.deepcopy(image)

        R=copy.deepcopy(sub_input)
        image_path=os.path.join(os.getcwd(),config.sample_dir)
        image_name='test_image'+str(ny*10+nx)+'.mat'
        image_path=os.path.join(image_path,image_name)
        scipy.io.savemat(image_path,{'R':R})

        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)
    '''

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

  '''
  if config.is_train:
    arrdata,arrlabel=mixSegment(arrdata,arrlabel)
  '''

  make_data(sess, arrdata, arrlabel)
  '''
  if not config.is_train:
    return nx, ny
  '''

def imsave(image, path):
  return scipy.misc.imsave(path, image)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img
