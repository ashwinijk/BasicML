#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import numpy as np
from numpy import newaxis
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from tensorflow.python import keras

# read a data file into the python console
df = pd.read_csv("D:\cloudaeye\python_part\data.csv")

#To print all the features (columns)
count = 0;
for columns in df:
    count = count+1
    #print(columns)
print("number of features:", count)
print("number of feature instances", df.size)

# We are labeling the status with NORMAL error as 1 and all other errors as 0
for i in range(0,df.status.size):
  if df.status[i] == 'NORMAL':
    df.status[i] = 1
  else: df.status[i] = 0

# We are not considering timestamp
df = df.loc[:, df.columns != 'timestamp']

# status column is fed as lebels in a seperate column
df_train = df.loc[:, df.columns != 'status']

# create labels using status column
label = df.status
label=label.astype('int')

#Split training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, label, test_size = 0.20, random_state = 42)


# In[4]:


#Simple Neural Network
model = keras.Sequential()
model.add(Dense(1, kernel_initializer = 'normal', input_shape = X_train.shape[1:]))
model.compile(loss= 'BinaryCrossentropy',optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Multi-layer Neural Network
model = keras.Sequential()
model.add(Dense(1, kernel_initializer = 'normal', input_shape = X_train.shape[1:] ))
model.add(Dense(1, kernel_initializer = 'normal', activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss= 'BinaryCrossentropy',optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)


# In[115]:


# conver 2d dataframe to 3d array
X_train_ar = X_train.to_numpy() # RNN input is a 3d tensor and hence  
X_test_ar = X_test.to_numpy() # convert numpy 2d dataframe to array first

X_train_3d = X_train_ar[:,newaxis,:] # RNN input is a 3d tensor and hence
X_test_3d = X_test_ar[:,newaxis,:] #  convert 2d to 3d array using newaxis

print(X_train_3d.shape)
print(X_test_3d.shape)

print(y_train.shape)
print(y_test.shape)

# RNN
model = Sequential()
model.add(SimpleRNN(5, input_shape=(X_train_3d.shape[1:]), return_sequences=True))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_3d, y_train, epochs=30)

# LSTM
model = Sequential()
model.add(LSTM(5, input_shape=(X_train_3d.shape[1:]), return_sequences=True))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_3d, y_train, epochs=30)


# In[ ]:


#CNN

# convert 2d data frame to 4d array 
X_train_ar = X_train.to_numpy() 
X_test_ar = X_test.to_numpy() 

X_train_ar = np.float32(X_train_ar)
X_test_ar = np.float32(X_test_ar)

res1 = np.concatenate([X_train_ar[np.newaxis] for arr in X_train_ar])
res2 = np.concatenate([X_test_ar[np.newaxis] for arr in X_test_ar])
X_train_4d = res1[:,:,:,newaxis]
X_test_4d = res2[:,:,:,newaxis]

#CNN model
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train_4d.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(X_train_4d, y_train, epochs=2)


# In[ ]:




