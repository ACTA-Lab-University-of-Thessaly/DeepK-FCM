# DeepK-FCM
# coding: utf-8
# Created by Theotziol
# Import important libraries


# *todo automate the CNN construction with a class and a separate script
#  

    
import pickle
from similarities import *
from preprocessing import *
from fcm_libr import *

from tensorflow import keras
import matplotlib.pyplot as plt

import os


#read the picles with dictionairies
with open('saved_dictionary_train.pkl', 'rb') as f3:
    dic_train = pickle.load(f3)

with open('saved_dictionary_test.pkl', 'rb') as f4:
    dic_test = pickle.load(f4)


dic_min_max_train = dic_normalization(dic_train)
dic_min_max_test = dic_normalization(dic_test)


### lets take the keys (classes) and perform experiments with just 2 classes
dic_keys = list(dic_train.keys())
classes_to_experiment = [0,1]
# classes_to_experiment = [55,44]


# step 2 clustering
from sklearn.cluster import KMeans
kmeans_centroids = {}
for i in classes_to_experiment:
  kmeans = KMeans(n_clusters=3)
  array = dic_min_max_train[i].copy().reshape((np.array(dic_min_max_train[i]).shape[0], np.array(dic_min_max_train[i]).shape[-1]))
  kmeans.fit(array)
  kmeans_centroids[i] = kmeans.cluster_centers_


#step 3 similarities

#suggested formula by Iakovidis https://www.sciencedirect.com/science/article/pii/S0020025523000099?ref=pdf_download&fr=RR-2&rr=7ccded525939ef04
def calculate_similarity(feature_vector, centroid, total_centroids):
  denominator = np.sum([np.linalg.norm(feature_vector - i) for i in total_centroids])
  return 1 - (np.linalg.norm(feature_vector-centroid) / denominator)


import itertools
#slice the detections dictionairy in only two classes

# two_classes_dic = dict(itertools.islice(dic_min_max_train.items(), 2))
two_classes_dic = {}
two_classes_dic_test = {}
for i in classes_to_experiment:
  two_classes_dic[i] = dic_min_max_train[i].copy()
  two_classes_dic_test[i] = dic_min_max_test[i].copy()


exp = experts(two_classes_dic, kmeans_centroids)
exp_test = experts(two_classes_dic_test, kmeans_centroids)

dic = similarities_per_centroids(exp)

fuzzy_dic = create_fuzzy_sets(dic)

## Optional plot fuzzy sets
# for i in list(fuzzy_dic.keys()):
#   if i == 'range':
#     continue
#   else:
#     for centroid in list(fuzzy_dic[i].keys()):
#       plt.plot(fuzzy_dic['range'], fuzzy_dic[i][centroid], label = centroid)
#     plt.legend()
#     plt.xlabel('Universe of Discourse')
#     plt.ylabel('μ(x)')
#     #plt.title(i)
#     plt.savefig(f'{i}.png')
#     plt.show()

matrix1 = input_to_input_weights(dic, fuzzy_dic)
matrix2 = input_to_output_weights(dic, fuzzy_dic)

# The FCM matrix
matrix = add_input_output_fcm(matrix1 , matrix2)

# testing
test_df = test_to_df(exp_test, kmeans_centroids)
input_df, output_df = split_labels(test_df, -len(classes_to_experiment))

fcm_inference = []
results = []

## Inference (Decision-making)
for i in range(len(input_df)):
  fcm1 = fcm(input_df.iloc[i].to_numpy()[None, :], matrix.to_numpy())
  result = fcm1.inference(lambda x: sigmoid(x,5), verbose = -1)[0][-len(classes_to_experiment):].numpy()
  fcm_inference.append(fcm1.inference_process)
  results.append(result)

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
accuracy = []
labels = []
predictions = []
for i in range(len(results)):
  label = np.argmax(output_df.iloc[i].to_numpy())
  labels.append(label)
  prediction = np.argmax(results[i])
  predictions.append(prediction)
  if label != prediction:
    text = 'missclassification'
  else:
    text = ''
  print(f'{i}, Label : {label}, Prediction {prediction},   input {input_df.iloc[i].to_numpy()[:-len(classes_to_experiment)]}, {text}')
  if label == prediction:
    accuracy.append(1)
  else:
    accuracy.append(0)
print(f'\nAccuracy = {np.sum(accuracy)/ len(accuracy)}')
stats = precision_recall_fscore_support(labels, predictions)
print(f'precision class 0 {stats[0][0]},\nprecision class 1 {stats[0][1]},\nrecall class 0 {stats[1][0]},\nrecall class 1 {stats[1][1]},\n')
df_conf = pd.DataFrame(confusion_matrix(labels, predictions), index = ['class 0', 'class 1'],columns = ['class 0', 'class 1'])
print(f'confusion matrix:\n{df_conf}')
print(f'f1_score = {f1_score(labels, predictions)}')
