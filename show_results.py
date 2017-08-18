import utils
import matplotlib.pyplot as plt
import os
import glob
import scipy.io
import copy
import numpy as np

dataset="sample"
data_dir=os.path.join(os.getcwd(),dataset)
data = glob.glob(os.path.join(data_dir,"*.mat"))
plt.ion()
plt.show()
N=len(data)
plt.figure(1)
results = {}
for i in range(N):
    imgg=scipy.io.loadmat(data[i])
    image=np.squeeze(imgg['R'])
    results[i] = image
    plt.subplot(211+i)
    _,name = os.path.split(data[i])
    plt.title(name)
    plt.imshow(image)
    plt.draw()
    plt.pause(5)
res=results[0]-results[1]
print results[0].max()
print results[1].max()
print res
print res.max()
