#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import cv2
import pickle
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Activation, Flatten, Concatenate, MaxPooling2D, Dense, concatenate, Dropout
from tensorflow.keras.models import Model


# In[2]:


PATH = "C:\\Users\\Luka\\Dropbox\\My PC (DESKTOP-E57AO1D)\\Desktop\\Dusan pollen\\projekat\\pollen\\train\\train"


# In[3]:


#Only once


# In[4]:


keys = ['scat', 'spec', 'life', 'labels']
data = {}
data2 = {}
sizes = {}
for key in keys:
    infile = open(PATH + '\\data_' + key +'.pkl','rb')
    data[key] = pickle.load(infile)
    infile.close()


# In[5]:


for key in keys:

    if key == 'labels':
        continue

    sizes[key] = data[key][0].shape[1]


# In[6]:


for key in keys:
    if key == 'labels':
        continue

    data2[key] = np.zeros((22200,sizes[key],sizes[key]))


# In[7]:


for i in range(22200):
    for key in keys:
        if key == 'labels':
            continue

        data2[key][i] = cv2.resize(data[key][i], (sizes[key],sizes[key]), interpolation = cv2.INTER_AREA)


# In[8]:


for key in keys:
    if key == 'labels':
        continue

    open_file = open(PATH + '\\data_' + key +'2.pkl','wb')
    pickle.dump(data2[key], open_file)
    open_file.close()


# In[9]:


#Import data


# In[10]:


for key in keys:
    if key == 'labels':
        infile = open(PATH + '\\data_' + key +'.pkl','rb')
    else:
        infile = open(PATH + '\\data_' + key +'2.pkl','rb')
    data[key] = pickle.load(infile)
    infile.close()


# In[12]:


labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(data['labels'])
labels = labelEncoder.transform(data['labels'])
labels = to_categorical(labels)


# In[13]:


data_train = {}
data_validation = {}
data_train['scat'], data_validation['scat'],data_train['spec'], data_validation['spec'],data_train['life'], data_validation['life'],labels_train, labels_validation = train_test_split(data['scat'],
                                                   data['spec'],
                                                   data['life'],
                                                   labels)


# In[ ]:


#Build the model


# In[ ]:


# def combined_model():
#     InputScat = Input(shape=(120, 120, 1), name = "ScatteringImage")
#     InputSpec = Input(shape=(32, 32, 1), name = "FluorescenceSpectrum")
#     InputLife = Input(shape=(24, 24, 1), name = "FluorescenceLifetime")

#     C1 = Conv2D(512, kernel_size = 9, padding='valid', activation='relu')(InputScat)
#     P1 = MaxPooling2D(pool_size=(2,2))(C1)
#     C2 = Conv2D(256, kernel_size = 7, padding='valid', activation='relu')(P1) 
#     P2 = MaxPooling2D(pool_size=(2,2))(C2)
#     C3 = Conv2D(128, kernel_size = 5, padding='valid', activation='relu')(P2) 
#     P3 = MaxPooling2D(pool_size=(2,2))(C3)
#     C4 = Conv2D(64, kernel_size = 3, padding='valid', activation='relu')(P3) 
#     P4 = MaxPooling2D(pool_size=(2,2))(C4)
#     flat = Flatten()(P4)
#     H1 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)

#     C1 = Conv2D(512, kernel_size= 7, padding='valid', activation='relu')(InputSpec)
#     P1 = MaxPooling2D(pool_size=(2,2))(C1)
#     C2 = Conv2D(256, kernel_size= 5, padding='valid', activation='relu')(P1)
#     P2 = MaxPooling2D(pool_size=(2,2))(C2) 
#     C3 = Conv2D(128, kernel_size = 3, padding='valid', activation='relu')(P2) 
#     P3 = MaxPooling2D(pool_size=(2,2))(C3)
#     flat = Flatten()(P3)
#     H2 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)

#     C1 = Conv2D(512, kernel_size= 7, padding='valid', activation='relu')(InputLife)
#     P1 = MaxPooling2D(pool_size=(2,2))(C1)
#     C2 = Conv2D(256, kernel_size= 5, padding='valid', activation='relu')(P1)
#     P2 = MaxPooling2D(pool_size=(2,2))(C2) 
#     C3 = Conv2D(128, kernel_size = 3, padding='valid', activation='relu')(P2) 
#     P3 = MaxPooling2D(pool_size=(2,2))(C3)
#     flat = Flatten()(P3)
#     H3 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)

#     fe = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate([H1, H2, H3]))
#     output = Dense(8, activation='softmax')(fe)
#     d_model = Model([a, b, c], output, name = "d_model")
#     d_model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return d_model


# In[ ]:


# def combined_model():
#     InputScat = Input(shape=(120, 120, 1), name = "ScatteringImage") #dopuna do kvadrata
#     InputSpec = Input(shape=(32, 32, 1), name = "FluorescenceSpectrum")
#     InputLife = Input(shape=(24, 24, 1), name = "FluorescenceLifetime")
    
    
#     C1 = Conv2D(256, kernel_size = 7, padding='valid', activation='relu')(InputScat)
#     P1 = MaxPooling2D(pool_size=(2,2))(C1)
#     C2 = Conv2D(128, kernel_size = 5, padding='valid', activation='relu')(P1) 
#     P2 = MaxPooling2D(pool_size=(2,2))(C2)
#     flat = Flatten()(P2)
#     H1 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)
    

#     C1 = Conv2D(256, kernel_size=5, padding='valid', activation='relu')(InputSpec)
#     P1 = MaxPooling2D(pool_size=(2,2))(C1)
#     C2 = Conv2D(128, kernel_size=3, padding='valid', activation='relu')(P1)
#     P2 = MaxPooling2D(pool_size=(2,2))(C2) 
#     flat = Flatten()(P2)
#     H2 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)
    
    
#     C1 = Conv2D(256, kernel_size=5, padding='valid', activation='relu')(InputLife)
#     P1 = MaxPooling2D(pool_size=(2,2))(C1)
#     C2 = Conv2D(128, kernel_size=3, padding='valid', activation='relu')(P1)
#     P2 = MaxPooling2D(pool_size=(2,2))(C2) 
#     C3 = Conv2D(64, kernel_size = 3, padding='valid', activation='relu')(P2) 
#     P3 = MaxPooling2D(pool_size=(2,2))(C3)
#     flat = Flatten()(P3)
#     H3 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)
    

#     fe = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate([H1, H2, H3]))
#     output = Dense(8, activation='softmax')(fe)
#     d_model = Model([InputScat, InputSpec, InputLife], output, name = "d_model")
#     d_model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return d_model


# In[31]:


def combined_model():
    Input_Scat = Input(shape=(120, 120, 1), name = "ScatteringImage")
    Input_Spec = Input(shape=(32, 32, 1), name = "FluorescenceSpectrum")
    Input_Life = Input(shape=(24, 24, 1), name = "FluorescenceLifetime")

    C1 = Conv2D(256, kernel_size = 7, padding='valid', activation='relu')(Input_Scat)
    P1 = MaxPooling2D(pool_size=(2,2))(C1)
    C2 = Conv2D(128, kernel_size = 5, padding='valid', activation='relu')(P1) 
    P2 = MaxPooling2D(pool_size=(2,2))(C2)
    C3 = Conv2D(64, kernel_size = 3, padding='valid', activation='relu')(P2) 
    P3 = MaxPooling2D(pool_size=(2,2))(C3)
    C4 = Conv2D(32, kernel_size = 3, padding='valid', activation='relu')(P3) 
    P4 = MaxPooling2D(pool_size=(2,2))(C4)
    flat = Flatten()(P4)
    H1 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)

    C1 = Conv2D(256, kernel_size= 5, padding='valid', activation='relu')(Input_Spec)
    P1 = MaxPooling2D(pool_size=(2,2))(C1)
    C2 = Conv2D(128, kernel_size= 3, padding='valid', activation='relu')(P1)
    P2 = MaxPooling2D(pool_size=(2,2))(C2) 
    C3 = Conv2D(64, kernel_size = 3, padding='valid', activation='relu')(P2) 
    P3 = MaxPooling2D(pool_size=(2,2))(C3)
    flat = Flatten()(P3)
    H2 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)

    C1 = Conv2D(256, kernel_size = 7, padding='valid', activation='relu')(Input_Life)
    P2 = MaxPooling2D(pool_size=(2,2))(C1)
    C2 = Conv2D(128, kernel_size = 5, padding='valid', activation='relu')(P1) 
    P2 = MaxPooling2D(pool_size=(2,2))(C2)
    flat = Flatten()(P2)
    H3 = Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')(flat)

    fe = Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(concatenate([H1, H2, H3]))
    output = Dense(8, activation='softmax')(fe)
    d_model = Model([Input_Scat, Input_Spec, Input_Life], output, name = "d_model")
    d_model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return d_model


# In[32]:


mod_el1 = combined_model()


# In[33]:


mod_el1.summary()


# In[34]:


#Train the model


# In[35]:


data_subset = {}
labels_subset = []

for key in keys:

    if key == 'labels':
        labels_validation = labels_train[:2850]
        labels_subset = labels_train[2850:]
    else:
        data_validation[key] = data_train[key][:2850]
        data_subset[key] = data_train[key][2850:]


# In[36]:


history_data = []
history_data_validation = []

for key in keys:

    if key == 'labels':
        continue

    history_data.append(data_subset[key])

for key in keys:

    if key == 'labels':
        continue

    history_data_validation.append(data_validation[key])


# In[40]:


history = mod_el1.fit(history_data, labels_subset,
                    epochs=9,
                    batch_size=16,
                    validation_data=(history_data_validation, labels_validation))


# In[41]:


mod_el1.save(PATH+'\\mod_el')


# In[ ]:




