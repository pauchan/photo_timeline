from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from scipy import misc
import numpy as np

def array_from_test_image(filename):
    inputImage = misc.imread(filename)
    reshapedImage = misc.imresize(inputImage,(32,32))

    if len(reshapedImage.shape) == 3:
        r = reshapedImage[:,:,0].flatten()
        g = reshapedImage[:,:,1].flatten()
        b = reshapedImage[:,:,2].flatten()
    else:
        r = reshapedImage[:,:].flatten()
        g = reshapedImage[:,:].flatten()
        b = reshapedImage[:,:].flatten()
        
    images = np.hstack((r,g,b))
     # consider .reshape for speed
    imagesArray = np.asarray(images)
    return images


model_path = 'cifar_grbm_smd.pkl'
model = serial.load(model_path)

X = model.get_input_space().make_theano_batch()
Y = model.mean_h_given_v(X)
Y2 = X*1
import pdb
pdb.set_trace()
   
f = function([X],Y2) # , on_unused_input='warn')

train_data = array_from_test_image('test.png')
y = f(train_data)
print y

