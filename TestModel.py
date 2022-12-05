#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Activation, Flatten, Concatenate, MaxPooling2D, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cv2


# In[2]:


DATA_SOURCE = "C:\\Users\\Luka\\Dropbox\\My PC (DESKTOP-E57AO1D)\\Desktop\\Dusan pollen\\projekat\\pollen\\train\\train\\"


# In[3]:


Modelnew = load_model(DATA_SOURCE+'\\mod_el')


# In[5]:


DATA_SOURCE_TEST = "C:\\Users\\Luka\\Dropbox\\My PC (DESKTOP-E57AO1D)\\Desktop\\Dusan pollen\\projekat\\pollen\\test\\New Folder\\"


# In[6]:


infile = open(DATA_SOURCE_TEST+'test_scat2.pkl','rb')
test_scat = pickle.load(infile)

infile = open(DATA_SOURCE_TEST+'test_spec2.pkl','rb')
test_spec = pickle.load(infile)

infile = open(DATA_SOURCE_TEST+'test_life2.pkl','rb')
test_life = pickle.load(infile)


# In[7]:


predictionsnew = Modelnew.predict([test_scat, test_spec, test_life])


# In[8]:


result = "ID,Category\n"
y = np.zeros(6169)
for i in range(6169):
  y[i] = np.argmax(predictionsnew[i])
  result = result + (str(i+1) + "," + str(int(y[i])) + "\n")
    
open_file = open(DATA_SOURCE_TEST+'\\predictionsnew.csv', "w")
open_file.write(result)


# In[ ]:




