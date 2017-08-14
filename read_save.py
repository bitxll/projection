import scipy

#glob.glob:获取可遍历的路径名
dataset = 'Train'
data_dir=os.path.join(os.getcwd(),dataset)

#read mat file
data=glob.glob(os.path.join(data_dir,"*.mat"))
imgg=scipy.io.loadmat(data[i])
image=imgg['R']

#read bmp file
data=glob.glob(os.path.join(data_dir,"*.bmp"))

