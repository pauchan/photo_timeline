# trying to get the same data structure as in classic image datasets
# get all the images from the data folder,resize to 32x32,  convert to numpy  and apply proper tags
# distribute them randomly, hopefully one batch file will be enough
# pack them to the bath file

#import PIL
#from PIL import Image
from scipy import misc
import pickle
import time
import os
import glob
import numpy as np
import pdb

#vstack

oldImageSize = 75

# stolen from
# https://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir-python
# to prevent iteration over hidden files
def listdir_nohidden(path):
    return glob.glob(os.path.join(path,'*'))

images = []
labels = []
#imagesArray = np.empty((0,3072),uint)

for idx,year_directory in enumerate(listdir_nohidden('trainingimages')):
    start_time = time.time()
    print year_directory
    for image_file in listdir_nohidden(year_directory):
#        print image_file
        inputImage = misc.imread(image_file)
        reshapedImage = misc.imresize(inputImage,(32,32))
        #test = np.hstack([reshapedImage[:,:,0].flatten(),reshapedImage[:,:,1].flatten(),reshapedImage[:,:,2].flatten()])
        #pdb.set_trace()
        # flattening the array to the 1024x3 rgb vector

        print image_file
        if len(reshapedImage.shape) == 3:
            r = reshapedImage[:,:,0].flatten()
            g = reshapedImage[:,:,1].flatten()
            b = reshapedImage[:,:,2].flatten()
        else:
            r = reshapedImage[:,:].flatten()
            g = reshapedImage[:,:].flatten()
            b = reshapedImage[:,:].flatten()
        
        
        images.append(np.hstack((r,g,b)))

        labels.append(idx)
    stop_time = time.time()
    print 'year {} processed with time {}'.format(year_directory,stop_time-start_time)

# consider .reshape for speed
imagesArray = np.asarray(images)
dataDictionary = {'data': imagesArray, 'labels': labels}

#pdb.set_trace()

pickle.dump(dataDictionary,open('data_batch','wb'))

# create labels dictionary

labelsList = range(1900,2000)
labelsDict = {'label_names': labelsList}
pickle.dump(labelsDict, open('batches_meta','wb'))


        


