from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def get_data(actors, filepath):
    testfile = urllib.URLopener()
    
    for a in actors:
        name = a.split()[1].lower()
        i = 0
        for line in open("facescrub_actors.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], "uncropped/male/" + filename), {}, 15)

                if not os.path.isfile("uncropped/male/"+filename):
                    continue
                try:
                    image = imread("uncropped/male/" + filename) #getimage
                    greyscale = rgb2gray(image) #turn into greyscale
                    dimensions = line.split()[5].split(',') #get dimensions
                    x1, y1 = int(dimensions[0]), int(dimensions[1])
                    x2, y2 = int(dimensions[2]), int(dimensions[3])
                    cropped_greyscale = imresize(greyscale[y1:y2, x1:x2], (32, 32)) #crop
                    imsave(filepath + filename, cropped_greyscale) #save
                    
                    i += 1
                    print "Fetching: {}".format(filename)
                    os.remove("uncropped/male/" + filename)
                except:
                    print "Error: {}".format(filename)

    for a in actors:
        name = a.split()[1].lower()
        i = 0
        for line in open("facescrub_actresses.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], "uncropped/female/"+filename), {}, 15)

                if not os.path.isfile("uncropped/female/"+filename):
                    continue
                try:
                    image = imread("uncropped/female/"+filename) #getimage
                    greyscale = rgb2gray(image) #turn into greyscale
                    dimensions = line.split()[5].split(',') #get dimensions
                    x1, y1 = int(dimensions[0]), int(dimensions[1])
                    x2, y2 = int(dimensions[2]), int(dimensions[3])
                    cropped_greyscale = imresize(greyscale[y1:y2, x1:x2], (32, 32)) #crop
                    imsave(filepath + filename, cropped_greyscale) #save
                    
                    i += 1
                    print "Fetching: {}".format(filename)
                    os.remove("uncropped/female/" + filename)
                except:
                    print "Error: {}".format(filename)

def get_sets(actor_data, labels, image_num, train_size, valid_size, test_size):    
    #Where actor_data are x values and labels are y values
    training_set = []
    validation_set = []
    test_set = []
    
    ty = []
    vy = []
    tey = []
    
    add = 0;
    for j in range(len(image_num)):
        image_index = np.arange(image_num[j])
        np.random.shuffle(image_index) #random shuffle
        
        for i in range(len(image_index)):
            image_index[i] += add
        
        training_set.extend([actor_data[i] for i in image_index[0:train_size]])
        validation_set.extend([actor_data[i] for i in image_index[train_size: train_size + valid_size]])
        test_set.extend([actor_data[i] for i in image_index[train_size+valid_size:train_size + valid_size + test_size]])
            
            
        ty.extend([labels[i] for i in image_index[0:train_size]])
        vy.extend([labels[i] for i in image_index[train_size:train_size + valid_size]])
        tey.extend([labels[i] for i in image_index[train_size+valid_size:train_size + valid_size + test_size]])
            
        add +=image_num[j]

    return np.array(training_set), np.array(ty), np.array(validation_set), np.array(vy), np.array(test_set), np.array(tey)
    
def get_image_data(actors, filepath):
    image_data = np.empty([0, 1024])
    labels = np.empty([0, len(actors)])
    image_count = []
    
    prev = -1;
    j = 0
    for i in actors: 

        if prev == -1:
            prev = actors.index(i)
            
        for file in os.listdir(filepath): 
            if i.split()[1].lower() in file:
                j+=1
                
                image = imread(filepath + file, flatten=True) 
                image_data = vstack((image_data, reshape(np.ndarray.flatten(image), [1, 1024])))
                
                label = np.zeros([1, len(actors)])
                label[0][actors.index(i)] = 1
                labels = vstack((labels, label))
                
            if actors.index(i) != prev:
                prev = actors.index(i)
                image_count.append(j)
                j = 0
            
    image_count.append(j)
    return image_data, labels, image_count
    
def get_image_data_gender(actors, gender, filepath):
    image_data = np.empty([0, 1024])
    labels = np.empty([0, 2])
    image_count = []
    
    prev = -1;
    j = 0
    for i in actors: 

        if prev == -1:
            prev = actors.index(i)
            
        for file in os.listdir(filepath): 
            if i.split()[1].lower() in file:
                j+=1
                
                image = imread(filepath + file, flatten=True) 
                image_data = vstack((image_data, reshape(np.ndarray.flatten(image), [1, 1024])))
                
                label = np.zeros([1, 2])
                if gender == 'm':
                    label[0][0] = 1
                else:
                    label[0][1] = 1
                labels = vstack((labels, label))
                
            if actors.index(i) != prev:
                prev = actors.index(i)
                image_count.append(j)
                j = 0
            
    image_count.append(j)    
    return image_data, labels, image_count