#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy.random import seed
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from collections import Counter
from tqdm import tqdm

import h5py
import numpy as np 
import os
import cv2
import random
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from keras.utils.np_utils import to_categorical
from PIL import Image



def loadDatasetOne():
    images = np.load("/Users/sumitkhare/braintumordatasetNPY/images.npy", allow_pickle=True)
    labels = np.load("/Users/sumitkhare/braintumordatasetNPY/labels.npy", allow_pickle=True)
    integer_to_class = {1: 'meningioma', 2: 'glioma', 3: 'pituitary tumor'}
    
    for i in range(0,len(labels)):
        if labels[i] == 1.0:
            labels[i] = 0
        if labels[i] == 2.0:
            labels[i] = 1
        if labels[i] == 3.0:
            labels[i] = 2
    return([images,labels])


def loadDatasetTwo():
    
    labelsa = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
    training_data = []
    training_labels = []
    
    for i in labelsa:
        folderPath = os.path.join('/Users/sumitkhare/newDataset/Training',i)
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath,j))
            training_data.append(img)
            training_labels.append(i)
        
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
   
    return([training_data, training_labels])
    
def rgb2Gray(images):
    #Define a new array to store new images in
    arr = []
    #loop through all images
    for i in images:
        #Using CV2 convert the image into grayscale
        img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        #Reshape the image
        img = img.reshape(256,256,1)
        #Store the image in the new array
        arr.append(img)
    #Return the array of converted images as numpy array
    return np.asarray(arr)


def resizeImages(images, dataset):
    #Empty array to store resized images in
    resizeImages = []
    
    #Loop through all images in dataset
    for i in images:
        #Assign each image to a new variable, use CV2 to resize the images and convert to float32 bit.
        new_img = cv2.resize(i, dsize=(256, 256),interpolation=cv2.INTER_CUBIC).astype(np.float32)
        #Convert the image to numpy array
        new_img = np.asarray(new_img)
        
        #Check if dataset is one, otherwise skip
        if dataset == 1:
            #set new dimentions to single channel
            new_img = new_img.reshape(256,256,1)
        #Append new image to array
        resizeImages.append(new_img)
    #Return the new array as a numpy array
    return np.asarray(resizeImages)

def reLabelOne(labels):
    newLabels = []
    for i in range(0,len(labels)):
        if labels[i] == 0:
            newLabels.append(1)
        if labels[i] == 1:
            newLabels.append(0)
        if labels[i] == 2:
            newLabels.append(3)
    return newLabels

def reLabelTwo(labels):
    for i in range(0,len(labels)):
        if labels[i] == 'glioma_tumor':
            labels[i] = 0
        if labels[i] == 'meningioma_tumor':
            labels[i] = 1
        if labels[i] == 'no_tumor':
            labels[i] = 2
        if labels[i] == 'pituitary_tumor':
            labels[i] = 3
    return labels


def augmentImages(image_array, label_array):
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.Affine(
            rotate=(-20, 20),
            shear=(-3, 3))
    ])
    
    augmented_images = seq(images=image_array)
    
    return [augmented_images, label_array]

def rotateDatasetOne(image_array, label_array):
    
    seq = iaa.Sequential([
        iaa.Affine(
            rotate=(90)),
    ])
    
    augmented_images = seq(images=image_array)
    
    return [augmented_images, label_array]


def loadNumpyDataset(dataSet):
    try:
        returnData = np.load(dataSet, allow_pickle=True)
        print(f'Dataset: {dataSet} loaded successfully')
        return returnData
    except Exception as e: 
        print(f'Dataset Failed to load: \n{e}')


# In[ ]:




