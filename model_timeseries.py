import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import gridspec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import  Flatten ,LSTM, Dense, RepeatVector,TimeDistributed, Conv1D, MaxPool1D, Add, Concatenate, Input, Dropout, Cropping1D, Conv1DTranspose
from copy import deepcopy as dc
import time
from sklearn.model_selection import KFold
import random
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

import os

import pickle
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M")

os.mkdir(dt_string)



def oversampling(train_x, train_y, label=1):
    '''
    function to perform oversampling in the training datasets by copying the minority indexes. 
    Args:
      train_x : 2D numpy array of the input (x) 
      train_y : 1D numpy array of the output (y)
      label : integer. The minority class label
    Returns:
     tuple : (oversampled (shuffled) input array, corresponing labels array)
    '''
    print(f'Oversampling class {label}...\n')
    #select the class
    indexes = np.where(train_y==label)[0]
    #copy_paste
    to_oversample = train_x[indexes].copy()
    new_labels = np.full(len(to_oversample), label)
    new_x = np.concatenate([train_x, to_oversample], axis = 0)
    new_y = np.concatenate([train_y, new_labels], axis = 0)
    #shuffle
    idx = np.random.permutation(len(new_x))
    x_shuffled,y_shuffled = new_x[idx], new_y[idx]
    return x_shuffled,y_shuffled


def save_dic_to_pickle(dic, dataset_name):
  '''
  save the predictions of the CNN model (from the last dense layer) into a pickle. 
  Saved predictions would allow experimentation with the clustering and the FCM parameters without re-using the CNN predictions, 
  thus saving time.
  '''
  with open(os.path.join(dt_string, 'saved_dictionary_{}.pkl'.format(dataset_name)), 'wb') as f:
    pickle.dump(dic, f)

# def preds_to_dic(dic, x, y, model):
#   for i, img in enumerate(x):
#     prediction = model.predict(img[None, :], verbose = 1)
#     dic[int(y[i])].append(prediction)
#   return dic

def preds_to_dic(dic, preds_x, y, model):
  '''
  save the predictions of the CNN model (from the last dense layer) into a dic. 
  Saved predictions would allow experimentation with the clustering and the FCM parameters without re-using the CNN predictions, 
  thus saving time.
  '''
  for i, img in enumerate(preds_x):
    dic[int(y[i])].append(img)
  return dic



def cnn_model(shape = (201, 3)):
    inpt = Input(shape = shape)
    cnn1 = Conv1D(64, 7, strides = 3, padding = 'valid',activation='relu')(inpt)
    pool1 = MaxPool1D()(cnn1)
    cnn2 = Conv1D(64, 5, strides = 2, padding = 'valid',activation='relu')(pool1)
    pool2 = MaxPool1D()(cnn2)
    cnn3 = Conv1D(64, 3, strides = 1, padding = 'valid',activation='relu')(pool2)
    global_max = keras.layers.GlobalMaxPooling1D()(cnn3)
    drop1 = Dropout(0.2)(global_max)
    dense = Dense(32, activation = 'sigmoid', name = 'dense_vector')(drop1)
    drop2 = Dropout(0.2)(dense)
    out = Dense(1, activation = 'sigmoid')(drop2)
    model = Model(inpt, out)
    metrics = [keras.metrics.BinaryAccuracy(),keras.metrics.FalseNegatives(name = 'fn'), keras.metrics.FalsePositives(name = 'fp'), 
                keras.metrics.TrueNegatives(name = 'tn'), keras.metrics.TruePositives(name = 'tp'),
                keras.metrics.AUC(name='prc', curve='PR')]
    model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = metrics)
    print(model.summary())
    return model


    
### The data to be loaded, change according to your data
new_array = np.load('all_sequences.npy')
y = np.load('labels_0_healthy.npy')

idx = np.random.permutation(len(new_array))
x_shuffled,y_shuffled = new_array[idx], y[idx]

print('Dataset split:\n')


train_x = x_shuffled[:int(len(x_shuffled)*0.9)]
test_x = x_shuffled[int(len(x_shuffled)*0.9):]

train_y = y_shuffled[:int(len(x_shuffled)*0.9)]
test_y = y_shuffled[int(len(x_shuffled)*0.9):]


np.save(os.path.join(dt_string, 'indexes.npy'), idx)
train_x, train_y = oversampling(train_x, train_y)

print(f'Training:\nhealthy curves = {len(train_y[train_y==0])}, defective curves = {len(train_y[train_y==1])}\n')
print(f'Testing:\nhealthy curves = {len(test_y[test_y==0])}, defective curves = {len(test_y[test_y==1])}\n')
print('Indexes saved in the file indexes.npy for experiment repeatability\n')


#Define a CNN model
model = cnn_model()

patience = 10
batch_size = 256
earlystopping = keras.callbacks.EarlyStopping(patience = patience,
                                                    monitor = 'val_loss',
                                                    restore_best_weights= True,
                                                    )
dic = {
                0 : 1,
                1 : 2
            }
# Train
history = model.fit(
  train_x, train_y,
  validation_split = 0.2,
  epochs=150,
  batch_size = batch_size,
  callbacks = [earlystopping],
  class_weight = dic 
)
# Evaluate 
vals1 = model.evaluate(test_x, test_y)

confusion_matrix = pd.DataFrame([[vals1[4], vals1[3]], [vals1[2], vals1[5]]],
 index = ['class 0', 'class 1'],
  columns=['class 0', 'class 1'])

print(f'confusion matrix:\n{confusion_matrix}')
# Store the results
confusion_matrix.to_csv(os.path.join( dt_string,'conf_matrix.csv'))
with open(os.path.join(dt_string, 'metrics.txt'), 'w') as f:
    f.write(f'CNN loss = {vals1[0]}\n')
    f.write(f'CNN Accuracy = {vals1[1]}\n')
    f.write(f'CNN AUC = {vals1[-1]}\n')
    f.write(f"Total epochs = {len(history.history['val_loss'])}, best epoch at {len(history.history['val_loss'])-patience}\n")
    f.write(f'Batch size = {batch_size}\n\n')
    f.write(f'Training:\nhealthy curves = {len(train_y[train_y==0])}, defective curves = {len(train_y[train_y==1])}\n')
    f.write(f'Testing:\nhealthy curves = {len(test_y[test_y==0])}, defective curves = {len(test_y[test_y==1])}\n')

# Acquire predictions of the last dense layer
new_model = keras.Model(model.input, model.get_layer('dense_vector').output)
train_dic = {}
test_dic = {}
train_dic[0] = []
train_dic[1] = []
test_dic[0] = []
test_dic[1] = []

print('\nPredict on train dataset...')
predictions_train = new_model.predict(train_x)
print('\nPredict on test dataset...')
predictions_test = new_model.predict(test_x)

# Save predictions to pickle and dictionairies.
dic2 = preds_to_dic(train_dic, predictions_train, train_y, new_model)
dic = preds_to_dic(test_dic, predictions_test,test_y, new_model)
save_dic_to_pickle(dic2, 'train')
save_dic_to_pickle(dic, 'test')

model.save(os.path.join(dt_string, 'classifier.h5'))
new_model.save(os.path.join(dt_string, 'extractor.h5'))
save_dic_to_pickle(dic2, 'train')
save_dic_to_pickle(dic, 'test')
print(f'Finished ...\nAll files were saved to {dt_string} folder')
