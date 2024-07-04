# DeepK-FCM

DeepK-FCM is a binary classification tool applicable to timeseries and imagery data. It employs Deep Learning (Convolutional Neural Networks), Clustering algorithms (tested with K-means) and Fuzzy Cognitive Maps. 
It has been succesfully applied in Antenna manufacturing quality inspection. More information regarding the methodology can be found in the authors' publication https://www.sciencedirect.com/science/article/pii/S1877050924000103.

The whole methodology was implemented in python language with the use of open-sourced libraries such as tensorflow. Check the requirements.txt to install the employed libraries. 
## Timeseries
Initially the CNN script (model_timeseries.py) is used to train and assess a CNN model and to store the predictions (training and testing predictions) from the last dense layer into pickles (as python dictionairies). 

Once the predictions have been stored, the main.py script is used to read the dictionairies, and to perform the next seps of clustering (K-means) and Fuzzy Cognitive Map Construction. 
Finally the decision-making (binary classification is performed.

Replace lines in the code containing custom data paths with your own data paths to use this package. 

## Image data

The whole preccesing pipeline is in the CNN_model_image.py script. Initially pre-trained State-of-the-Art 2D CNNs are employed and assessed with transfer learning (and custom data). Once the models have been trained, the final dense layer is employed for predictions. 
The training predictions are used to construct an FCM with K-means clustering and similarities between cluster centers. Finally decision-making is performed with the DeepK-FCM. 

Replace lines in the code containing custom data paths with your own data paths to use this package.


