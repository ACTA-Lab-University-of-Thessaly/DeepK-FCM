import numpy as np 
from sklearn.cluster import KMeans
import skfuzzy
from skfuzzy.membership import trimf
import time
import pandas as pd
from fcm_libr import *
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score

#step1 preprocessing
def dic_normalization(dic):
    dic_min_max = {}
    for clas in list(dic.keys()):

        dic_min_max[clas] = []

        for vctr in range(len(dic[clas])):
            mx = np.max(dic[clas][vctr])
            mn = np.min(dic[clas][vctr])

            f_min_max = lambda x: (x - mn) / (mx - mn)

            dic_min_max[clas].append(f_min_max( dic[clas][vctr] ))

        dic_min_max[clas] = np.asarray(dic_min_max[clas])
    return dic_min_max

#step2 concepts definition and similarity
def kmeans(dic_min_max_train, classes_to_experiment = [0,1], n_clusters = 2):
    kmeans_centroids = {}
    for i in classes_to_experiment:
        kmeans = KMeans(n_clusters=n_clusters)
        array = dic_min_max_train[i].copy().reshape((np.array(dic_min_max_train[i]).shape[0], np.array(dic_min_max_train[i]).shape[-1]))
        kmeans.fit(array)
        kmeans_centroids[i] = kmeans.cluster_centers_
    return kmeans_centroids

def calculate_similarity(feature_vector, centroid, total_centroids):
    #suggested formula by Iakovidis https://www.sciencedirect.com/science/article/pii/S0020025523000099?ref=pdf_download&fr=RR-2&rr=7ccded525939ef04
    denominator = np.sum([np.linalg.norm(feature_vector - i) for i in total_centroids])
    return 1 - (np.linalg.norm(feature_vector-centroid) / denominator)


def experts(dic_images, dic_centroids):
    '''
    dic_images : a dictionairy of {class_0: [feature_vectors_0,... feature_vectors_n], ... class_y: [feature_vectors_0,... feature_vectors_n]}
    dic_centroids : a dictionairy of {class_0: [centroid_0,... centroid_m], ... class_y: [centroid_0,... centroid_m]}

    Returns:
        A dictionairy of {
        class_0: {feature_vectors_index_0 : [similarity_centroid_0, ... similarity_centroid_m], feature_vectors_index_n : [similarity_centroid_0, ... similarity_centroid_m]},
        ... ,
        class_y: {feature_vectors_index_0 : [similarity_centroid_0, ... similarity_centroid_m], feature_vectors_index_n : [similarity_centroid_0, ... similarity_centroid_m]}}
        }
    '''
    total_centroids_denominator = []
    for clas in list(dic_centroids.keys()):
        for cntrd in range(len(dic_centroids[clas])):
            total_centroids_denominator.append(dic_centroids[clas][cntrd])
    experts = {}
    for clas in list(dic_images.keys()):
        experts[clas] = {}
        for f_v in range(len(dic_images[clas])):
            experts[clas][f_v] = []
            for cl in list(dic_centroids.keys()):
                for cntrd in range(len(dic_centroids[clas])):
                    experts[clas][f_v].append(
                        np.round(calculate_similarity(dic_images[clas][f_v], dic_centroids[cl][cntrd], total_centroids_denominator),2)
                    )

    return experts



def similarities_per_centroids(exp_dic):
    '''
    Function to convert the expert dictionairy to centroids dictionairy
    Args:
        exp_dic: A dic of {class_1 : expert : [centroid_1_1 similarity, centroid_1_2 similarity, .... centroid_z_zm similarity] ,
                                    ...
                            class_z : expert : [centroid_1_1 similarity, centroid_1_2 similarity, .... centroid_z_zm similarity] ...}

    Returns:
        dic: A dictionairy {centroid_nameOfClass_num : [similarity_1, ... similarity_n], ...
                            centroid_nameOfClass_num : [similarity_1, ... similarity_n]}

    '''
    classes = list(exp_dic.keys())
    total_centroids = len(exp_dic[classes[0]][0])

    name = 'centroid_'
    array = []
    for clas in classes:
        for expert in list(exp_dic[clas].keys()):
            array.append(exp_dic[clas][expert])
    array = np.array(array)
    dic = {}
    start = 0
    for i in classes:
        for j in range(total_centroids//len(classes)):
            dic[name + str(i) + '_' + str(j)] = array[:, start]
            start +=1
    return dic

def create_fuzzy_sets(dic, n_sets = 3, names = ['low', 'medium', 'high']):
    '''
    Create a dic that contains the fuzzy sets for each centroid

    Args:
        dic : A dictionairy {centroid_nameOfClass_num : [similarity_1, ... similarity_n], ...
                            centroid_nameOfClass_num : [similarity_1, ... similarity_n]}
        n_sets : the number of the fuzzy membership functions (predefined 3, DONT CHANGE)
        names : the names of fuzzy membership fanctions

    Returns:
        fuzzy_dic : a dictionairy {range : [0, 1],
                                centroid_class_num : np.array( fuzzy membership)}
    '''
    assert n_sets == len(names)
    rnge = np.arange(0.0, 1.1, 0.01)
    fuzzy_dic = {}
    fuzzy_dic['range'] = rnge
    for centroid in list(dic.keys()):
        minimum = dic[centroid].min()
        maximum = dic[centroid].max()
        mean = np.round(dic[centroid].mean(), 2)
        fuzzy_dic[centroid] = {}
        for nm in names:
            if nm == 'low':
                fuzzy_dic[centroid][nm] = trimf(rnge, [rnge.min(), minimum, mean])
            elif nm == 'medium':
                fuzzy_dic[centroid][nm] = trimf(rnge, [minimum, mean, maximum])
            else:
                fuzzy_dic[centroid][nm] = trimf(rnge, [mean, maximum, 1.0])
    return fuzzy_dic


#Step 3 matrix definition

def input_to_input_weights(dic, fuzzy_dic):
    fcm = return_pandas_df_zeros(dic)
    input_concepts = list(dic.keys()) #centroids
    for column in input_concepts:
        for row in input_concepts:
            if row == column:
                continue
            else:
                similarities_i = []
                similarities_j = []
                m_i = []
                m_j = []

                for i in dic[row]:
                    m_values = find_membership_values(fuzzy_dic, i, row)
                    m = m_values.max() #take the membership where belongs
                    similarities_i.append(i)
                    m_i.append(m)

                for j in dic[column]:
                    m_values = find_membership_values(fuzzy_dic, j, column)
                    m = m_values.max() #take the membership where belongs
                    similarities_j.append(j)
                    m_j.append(m)
                similarities_i = np.array(similarities_i)
                similarities_j = np.array(similarities_j)
                m_i = np.array(m_i)
                m_j = np.array(m_j)

                fcm[column][row] =  (np.sum(similarities_i * m_i) + np.sum(similarities_j * m_j)) / (np.sum(m_i) + np.sum(m_j))
                fcm[row][column] =  (np.sum(similarities_i * m_i) + np.sum(similarities_j * m_j)) / (np.sum(m_i) + np.sum(m_j))
    return fcm

def input_to_output_weights(dic, fuzzy_dic):
    fcm = return_pandas_df_zeros(dic)
    input_concepts = list(dic.keys()) #centroids
    output_concepts = [i for i in fcm.columns if i not in input_concepts]
    # print(input_concepts, output_concepts)
    for column in output_concepts:
        for row in input_concepts:
            similarities_i = []
            m_i = []
            for i in dic[row]:
                m_values = find_membership_values(fuzzy_dic, i, row)
                m = m_values.max() #take the membership where belongs
                m_i.append(m)
                similarities_i.append(i)

            similarities_i = np.array(similarities_i)
            m_i = np.array(m_i)

            fcm[column][row] =  (np.sum(similarities_i * m_i)) / np.sum(m_i)
    return fcm

def find_membership_values(fuzzy_dic, value, centroid):
    '''
    Function to return the membership values for one centroid
    '''
    rnge = fuzzy_dic['range']
    index = np.where( np.abs(rnge - value) < 0.01)[0][0]
    m_values = []
    for m_f in list(fuzzy_dic[centroid].keys()):
        if fuzzy_dic[centroid][m_f][index] !=0:
            m_values.append(fuzzy_dic[centroid][m_f][index])
    return np.array(m_values)


def return_pandas_df_zeros(dic):
    centroids = list(dic.keys())
    classes = [i.split('_')[1] for i in centroids]
    unique_classes = []
    for i in classes:
        if i not in unique_classes:
            unique_classes.append(i)
    axis = len(centroids) + len(unique_classes)
    columns = []
    for i in centroids:
        columns.append(i)
    for i in unique_classes:
        columns.append(f'class_{i}')
    fcm = np.zeros((axis,axis))
    fcm_df = pd.DataFrame(fcm, index = columns, columns = columns)
    return fcm_df

def add_input_output_fcm(fcm_inp, fcm_otp):
  final_fcm = fcm_inp + fcm_otp
  columns = final_fcm.columns
  centroid = 'centroid'
  cls = 'class'
  for column in columns:
    for row in columns:
      cls_indx = 1
      if column.split('_')[cls_indx] != row.split('_')[cls_indx]:
        if final_fcm[column][row] != 0:
          final_fcm[column][row] = final_fcm[column][row] *(-1)
  return final_fcm

def define_matrix(dic, fuzzy_dic):
    '''
    dic: Dictionairy from the similarities_per_centroid function
    fuzzy_dic : Dictionairy from the create_fuzzy_sets function
    '''
    matrix1 = input_to_input_weights(dic, fuzzy_dic)
    matrix2 = input_to_output_weights(dic, fuzzy_dic)
    matrix = add_input_output_fcm(matrix1 , matrix2)
    return matrix


def test_to_df(exp_dic, centroids):
    total_classes = list(exp_dic.keys())
    classes = []
    similarities = []
    for i in total_classes:
        experts_len = len(exp_dic[i].keys())
        for exp in range(experts_len):
            classes.append(i)
            similarities.append(exp_dic[i][exp])

    total_classes_centroids = list(centroids.keys())
    columns = []
    for i in total_classes_centroids:
        for n in range(centroids[i].shape[0]):
            columns.append(f'class_{i}_{n}')
    df = pd.DataFrame(similarities, columns = columns)
    df['classes'] = classes
    for i in total_classes:
        df[f'class_{i}'] = np.zeros(len(df))
        df[f'class_{i}'].loc[df[f'classes'] == i] = 1

    df.pop('classes')
    return df

def split_labels(df, labels_index = -1):
    input_df = df.copy().sample(frac=1)
    df_labels = input_df[df.columns[labels_index:]]
    input_df[df.columns[labels_index:]] = 0.0
    return input_df, df_labels


def inference(input_df, output_df, matrix, l=1, classes_to_experiment=[0,1], n_iter=30):
    fcm_inference = []
    results = []
    for i in range(len(input_df)):
        fcm1 = fcm(input_df.iloc[i].to_numpy()[None, :], matrix.to_numpy(), n_iterations=n_iter)
        result = fcm1.inference(lambda x: sigmoid(x,l), verbose = -1)[0][-len(classes_to_experiment):].numpy()
        fcm_inference.append(fcm1.inference_process)
        results.append(result)
    return results, fcm_inference


def calculate_stats(results, output_df, classes_to_experiment=[0,1]):
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
        # print(f'{i}, Label : {label}, Prediction {prediction},   input {input_df.iloc[i].to_numpy()[:-len(classes_to_experiment)]}, {text}')
        if label == prediction:
            accuracy.append(1)
        else:
            accuracy.append(0)
    print(f'\nAccuracy = {np.sum(accuracy)/ len(accuracy)}')
    stats = precision_recall_fscore_support(labels, predictions)
    print(f'precision class 0 {stats[0][0]},\nprecision class 1 {stats[0][1]},\nrecall class 0 {stats[1][0]},\nrecall class 1 {stats[1][1]},\n')
    df_conf = pd.DataFrame(confusion_matrix(labels, predictions), index = ['class 0', 'class 1'],columns = ['class 0', 'class 1'])
    print(f'confusion matrix:\n{df_conf}')
    return np.sum(accuracy)/ len(accuracy), stats[0][1], stats[1][1], df_conf