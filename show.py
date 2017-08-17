import utils
import matplotlib.pyplot as plt
import os
import glob
import scipy.io
import copy
import numpy as np

dataset="train_data"
data_dir=os.path.join(os.getcwd(),dataset)
data = glob.glob(os.path.join(data_dir,"*.mat"))
sub_input_sequence = []
sub_label_sequence = []
n_steps=64
out_step=1
plt.ion()
plt.show()
imgg=scipy.io.loadmat(data[0])
image=imgg['R']
h,w=image.shape
for y in range(n_steps, (h-(out_step)),20):
    sub_input = copy.deepcopy(image)
    sub_input = image[(y-n_steps):y,:]
    print type(sub_input)
    sub_label = np.squeeze(image[y:(y+out_step),:])
    plt.imshow(sub_input)
    #plt.draw()
    plt.pause(5)
    sub_input_sequence.append(sub_input)
