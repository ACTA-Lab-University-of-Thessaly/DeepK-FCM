# DeepK-FCM
# coding: utf-8
# Created by Theotziol
# Import important libraries


# *todo automate the CNN construction with a class and a separate script
#  
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
from PIL import Image
#get_ipython().run_line_magic('matplotlib', 'inline')
import random
import glob
import time
from scipy import stats
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle
from similarities import *
from preprocessing import *
from fcm_libr import *
# from fcm_pso import *

def save_dic_to_pickle(dic,path, dataset_name):
        with open(os.path.join(path, 'saved_dictionary_{}.pkl'.format(dataset_name)), 'wb') as f:
            pickle.dump(dic, f)

def preds_to_dic(dic, array, labels, model):
  for i, img in enumerate(array):
    prediction = model.predict(img[None, :])
    label = int(labels[i])
    dic[label].append(prediction)
  return dic

def preds_to_dic_batched(dic, predictions, labels):
    for i in range(len(labels)):
        dic[int(labels[i])].append(predictions[i])
    return dic
    


# Define images path
path_def = "whole/defective/*"
path_healthy = "whole/healthy/*"

#create a folder for the statistics
from datetime import datetime
cwd = os.getcwd()
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M")

statistics_folder =  'vgg19-' + dt_string
os.mkdir(statistics_folder) 

 
# Read addresses and labels from the images' folder
addrs1 = glob.glob(path_def)
addrs2 = glob.glob(path_healthy)

addrs = addrs1 + addrs2
#len(addrs)

 
# Create labels' vector for each image in the dataset

y1 = [0 for i in addrs1]
y2 = [1 for i in addrs2]
y = y1 + y2
# Number of classes
classes_num = 1 
 
# Make a list of the images with the labels
c = list(zip(addrs, y))
# Shuffle the list
random.shuffle(c)
# Create tuples with the addresses and the labels
addrs, y = zip(*c)
 
# Set the image size variables
dimension = 256
  
 
# Create the function for the image preprocessing (BGR to grayscale)
def read_and_process_image(list_of_images):
    X=[]
    for img in addrs:
 
        #Show name of image
        #print(img)
 
        image = cv2.imread(img)
       
        #Insert to a list the image
        image = cv2.resize(image, (dimension,dimension), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X.append(image)
    return X
 
 
# Select number of channels
channels = 3  # 1 grayscale, 3 rgb
 
# Create the X, y datasets for the ML processing
X= read_and_process_image(addrs)
 
X = np.array(X)
y = np.array(y)
 
print(f'total images {X.shape}')
print(y.shape)

# Kfold
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet50V2 
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import vgg19

kf = KFold(n_splits = 5, shuffle = True)
fold = 1

accuracies_first = []
accuracies_second = []
auc_second = []
dic_stats = {}
dic_stats_ev = {}

for train_index, test_index in kf.split(X):

    path = os.path.join(cwd, statistics_folder, f'fold{fold}')
    os.mkdir(path)
    print(f"Fold {fold}\n")
    x_train, y_train = X[train_index], y[train_index]
    x_test, y_test = X[test_index], y[test_index]

    #create a validation dataset for training
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
 
    # Get the length of the train and validation data
    ntrain = len(x_train)
    nval = len(x_val)
    ntest = len(x_test)
    print("X train length:", len(x_train))
    print("X validation length:", len(x_val))
    print("X test length:", len(x_test))

    # Batch size (should be a factor of 2.***4,8,16,32,64...***)
    batch_size = 16

    # Dropout rate
    drop_rate = 0.2
    
    # Number of epochs
    num_epochs = 400

## IF NEEDED we perform data augmentation
# Set the transformations and augmentations for the training set (in cases of small datasets this helps)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(#rescale=1./255,   #Scale the image between 0 and 1
                                        # rotation_range=90,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.1,
                                        zoom_range=0.1,
                                        horizontal_flip=True,
                                        vertical_flip=True)
 
# rescale is part of the model based on the pretrained requirements
    val_datagen = ImageDataGenerator() #rescale=1./255)
    test_datagen = ImageDataGenerator() #rescale=1./255)

# Create the generators
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)
    
    test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
 
    

    base_model = VGG19(include_top = False,
                            weights = 'imagenet',
                            input_shape = (dimension, dimension, 3),
                            pooling = None)

    base_model.trainable = False
    inputs = keras.layers.Input(shape=(dimension, dimension, 3))
    x = keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation = 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation = 'relu')(x)
    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = keras.Model(inputs, outputs, name = 'vgg19')
    optimizer = keras.optimizers.Adam(learning_rate= 0.001)
    model.compile(optimizer=optimizer, loss='bce', 
                metrics = [keras.metrics.BinaryAccuracy(name = 'accuracy'),
                keras.metrics.Precision(name = 'precision'),
                keras.metrics.Recall(name = 'recall'), 
                keras.metrics.AUC(name='prc', curve='PR')] 
                )

# Model training
# Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    start = time.time()
    history = model.fit(train_generator,
                                    steps_per_epoch = ntrain // batch_size,
                                    epochs=num_epochs,
                                    validation_data = val_generator,
                                    validation_steps = ntest // batch_size,
                                    callbacks=[early_stopping])

    end = time.time()
    print('TIME = ', end-start, "sec.")
    # model.save(os.path.join(cwd, statistics_folder, f'ResNet101_fold_{fold}.h5'))

# Model evaluation
    val_generator.reset() # reset the generator to force it start from the begining
    eval_gen = np.round(model.evaluate(val_generator, steps = nval // batch_size, workers=1, use_multiprocessing=False), 5)
    print("*****************************")
    print("Evaluation accuracy and loss first stage")
    print(" accuracy =", eval_gen[1] )
    print(" loss =", eval_gen[0])
    print("*****************************")
    test_generator.reset() # reset the generator to force it start from the begining
    pred_gen = np.round(model.evaluate(test_generator, steps = ntest // batch_size, workers=1, use_multiprocessing=False), 5)
    print("Testing accuracy and loss first stage")
    print(" accuracy =", pred_gen[1])
    print(" loss =", pred_gen[0])
    print("*****************************")
    accuracies_first.append(pred_gen[1])

    test_generator.reset() # reset the generator to force it start from the begining
    # If the predict_generator doesn't work, try (nval//batch_size)+1

    predictions = model.predict(test_generator, steps =None, workers=1, use_multiprocessing=False)
    
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    # Separate Confusion Matrices
    y_true = y_test
    y_pred = predictions

    labels = ['defective', 'healthy']

# Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm_gen = confusion_matrix(y_test, predictions)
    print(cm_gen)

    import pandas as pd 
    matrix_df = pd.DataFrame(cm_gen, index = labels, columns = labels)
    matrix_df.to_csv(os.path.join(path, f'conf_matrix_first_stage.csv'))

    with open(os.path.join(path, f'stats_first_stage.txt'), 'w') as fl:
        fl.write(f"Evaluation accuracy first stage = {eval_gen[1] }\n" )
        fl.write(f"Evaluation loss first stage = = {eval_gen[0] }\n" )
        fl.write(f"Testing accuracy first stage = = {pred_gen[1]}\n" )
        fl.write(f"Testing loss first stage = = {pred_gen[0] }\n" )

    target_names = ['Defective', 'healthy']
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, predictions, target_names=target_names)
    print(cr)

    with open(os.path.join(path, f'report_first_stage.txt'), 'w') as fl:
        fl.write(cr)

    from sklearn.metrics import roc_auc_score

    from sklearn import  metrics

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, 'training_accuracy_first_stage.png'), dpi = 300)
    plt.show()


# Plot training & validation loss values
    plt.plot(history.history['loss'], color = 'g')
    plt.plot(history.history['val_loss'], color = 'b')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, 'training_loss_first_stage.png'), dpi = 300)
    plt.show()

    print("fine tuning\n\n")
    train_generator.reset()
    val_generator.reset()
    test_generator.reset()
    

    for layer in model.layers:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
    optimizer = keras.optimizers.Adam(learning_rate= 0.00001)
    model.compile(optimizer=optimizer, loss='bce', 
                metrics = [keras.metrics.BinaryAccuracy(name = 'accuracy'),
                keras.metrics.Precision(name = 'precision'),
                keras.metrics.Recall(name = 'recall'), 
                keras.metrics.AUC(name='prc', curve='PR')] 
                )
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    start = time.time()
    history = model.fit(train_generator,
                                    steps_per_epoch = ntrain // batch_size,
                                    epochs=num_epochs,
                                    validation_data = val_generator,
                                    validation_steps = ntest // batch_size,
                                    callbacks=[early_stopping])

    end = time.time()
    print('Fine tuning TIME = ', end-start, "sec.")

    model.save(os.path.join(path, 'vgg19_fine_tuned.keras'))

# Model evaluation
    val_generator.reset() # reset the generator to force it start from the begining
    eval_gen = np.round(model.evaluate(val_generator, steps = nval // batch_size, workers=1, use_multiprocessing=False), 5)
    print("*****************************")
    print("Evaluation accuracy and loss fine tuning")
    print(" accuracy =", eval_gen[1] )
    print(" loss =", eval_gen[0])
    print("*****************************")
    test_generator.reset() # reset the generator to force it start from the begining
    pred_gen = np.round(model.evaluate(test_generator, steps = ntest // batch_size, workers=1, use_multiprocessing=False), 5)
    print("Testing accuracy and loss fine tuning")
    print(" accuracy =", pred_gen[1])
    print(" loss =", pred_gen[0])
    print("*****************************")
    accuracies_second.append(pred_gen[1])
    auc_second.append(pred_gen[-1])

    test_generator.reset() # reset the generator to force it start from the begining
    # If the predict_generator doesn't work, try (nval//batch_size)+1

    predictions = model.predict(test_generator, steps =None, workers=1, use_multiprocessing=False)
    
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    # Separate Confusion Matrices
    y_true = y_test
    y_pred = predictions

    labels = ['defective', 'healthy']

# Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm_gen = confusion_matrix(y_test, predictions)
    print(cm_gen)

    import pandas as pd 
    matrix_df = pd.DataFrame(cm_gen, index = labels, columns = labels)
    matrix_df.to_csv(os.path.join(path, f'conf_matrix_fine_tuned.csv'))

    with open(os.path.join(path, f'stats_fine_tuned.txt'), 'w') as fl:
        fl.write(f"Evaluation accuracy fine tuned = {eval_gen[1] }\n" )
        fl.write(f"Evaluation loss fine tuned = = {eval_gen[0] }\n" )
        fl.write(f"Testing accuracy fine tuned = = {pred_gen[1]}\n" )
        fl.write(f"Testing loss fine tuned = = {pred_gen[0] }\n" )

    target_names = ['Defective', 'healthy']
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, predictions, target_names=target_names)
    print(cr)

    with open(os.path.join(path, f'report_fine_tuned.txt'), 'w') as fl:
        fl.write(cr)

    from sklearn.metrics import roc_auc_score

    from sklearn import  metrics

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, 'training_accuracy_fine_tuned.png'), dpi = 300)
    plt.show()


# Plot training & validation loss values
    plt.plot(history.history['loss'], color = 'g')
    plt.plot(history.history['val_loss'], color = 'b')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, 'training_loss_fine_tuned.png'), dpi = 300)
    plt.show()


    #deepk -FCM
    new_model = keras.models.Model(model.input, model.layers[-2].output)
    train_dic = {}
    test_dic = {}
    classes_to_experiment = [0,1]
    for i in classes_to_experiment:
        train_dic[i] = []
        test_dic[i] = []
    
    predictions_train = new_model.predict(x_train)
    predictions_val = new_model.predict(x_val)
    predictions_test = new_model.predict(x_test)

    train_dic = preds_to_dic_batched(train_dic, predictions_train, y_train)
    print(f'Predicted training dataset, nsamples {np.sum([len(train_dic[i]) for i in classes_to_experiment])}\n')
    train_dic = preds_to_dic_batched(train_dic, predictions_val, y_val)
    print(f'Predicted validation dataset, total samples {np.sum([len(train_dic[i]) for i in classes_to_experiment])}\n')
    test_dic = preds_to_dic_batched(test_dic, predictions_test, y_test)
    print(f'Predicted test dataset, total samples {np.sum([len(test_dic[i]) for i in classes_to_experiment])}\n')

    dic_min_max_train = dic_normalization(train_dic)
    dic_min_max_test = dic_normalization(test_dic)

    l_s = [1, 5]
    clusters = [2,3,4,5]
    dic_stats[fold] = {}
    dic_stats_ev[fold] = {}
    for l in l_s:
        dic_stats[fold][l] = {}
        dic_stats_ev[fold][l] = {}

        for c in clusters:
            kmeans_centroids = kmeans(dic_min_max_train, n_clusters=c)
            exp = experts(dic_min_max_train, kmeans_centroids)
            exp_test = experts(dic_min_max_test, kmeans_centroids)
            dic = similarities_per_centroids(exp)
            fuzzy_dic = create_fuzzy_sets(dic)

            matrix = define_matrix(dic, fuzzy_dic)
            test_df = test_to_df(exp_test, kmeans_centroids)
            input_df, output_df = split_labels(test_df, -len(classes_to_experiment))
            results, fcm_inferences = inference(input_df, output_df, matrix, l)
            stats = calculate_stats(results, output_df)
            with open(os.path.join(path, f'stats_deepk_l{l}_cl{c}.txt'), 'w') as fl:
                fl.write(f"Accuracy = {stats[0] }\n" )
                fl.write(f"Precision = {stats[1] }\n" )
                fl.write(f"Recall = {stats[2]}\n" )
            matrix.to_csv(os.path.join(path, f'deepkfcm_matrix_l{l}_cl{c}.csv'))
            stats[-1].to_csv(os.path.join(path, f'deepkfcm_conf_matrix_l{l}_cl{c}.csv'))
            dic_stats[fold][l][c] = [stats[0], stats[1], stats[2]]

            #deep evolutionairy
            n_iter = 5
            train_df = convert_dic_to_categorical_df(exp)
            test_df = convert_dic_to_categorical_df(exp_test)
            fcm_train_x, fcm_train_y = split_labels(train_df, -len(classes_to_experiment))
            fcm_test_x, fcm_test_y = split_labels(test_df, -len(classes_to_experiment))
            opt = pso(fitness_fn =fitness_function(),n_iter=80,pop_size=80,dim=(fcm_train_x.shape[-1],fcm_train_x.shape[-1] ), x_offset=0.1)
            
            for i in range(len(classes_to_experiment)):
                opt.specify_output_concept(-(i+1))
            opt.fcm_learning_classification(fcm_train_x.to_numpy(),fcm_train_y.to_numpy(), transfer_function = sigmoid, fcm_iterations=n_iter, l=l)

            matrix_ev = pd.DataFrame(opt.g.numpy(), columns = train_df.columns, index = train_df.columns)
            matrix_ev.to_csv(os.path.join(path, f'best_fcm_evolutionairy_l{l}_cl{c}.csv'))
            results_ev, fcm_inferences_ev = inference(fcm_test_x, fcm_test_y, matrix_ev, opt.l, n_iter=n_iter)
            stats_ev = calculate_stats(results_ev, fcm_test_y)
            with open(os.path.join(path, f'stats_deepk_evolution_l{l}_cl{c}.txt'), 'w') as fl:
                fl.write(f"Accuracy = {stats_ev[0] }\n" )
                fl.write(f"Precision = {stats_ev[1] }\n" )
                fl.write(f"Recall = {stats_ev[2]}\n" )
            stats_ev[-1].to_csv(os.path.join(path, f'deepk_evolution_conf_matrix_l{l}_cl{c}.csv'))
            dic_stats_ev[fold][l][c] = [stats_ev[0], stats_ev[1], stats_ev[2]]
    
    #finished fold
    fold += 1


print('Accuraccies in the first stage = \n', accuracies_first, '\n')
print(f'mean = {np.mean(accuracies_first)}, std = {np.std(accuracies_first)}\n')
print('Accuraccies after fine-tuning = \n', accuracies_second, '\n')
print(f'mean = {np.mean(accuracies_second)}, std = {np.std(accuracies_second)}\n')
print('AUC after fine-tuning = \n', auc_second, '\n')
print(f'mean = {np.mean(auc_second)}, std = {np.std(auc_second)}\n')



    


