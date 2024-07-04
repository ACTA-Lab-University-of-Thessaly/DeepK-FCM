# DeepK-FCM

DeepK-FCM is a binary classification tool applicable to timeseries and imagery data. It employs Deep Learning (Convolutional Neural Networks), Clustering algorithms (tested with K-means) and Fuzzy Cognitive Maps. 
It has been succesfully applied in Antenna manufacturing quality inspection. More information regarding the methodology can be found in the authors' publication https://www.sciencedirect.com/science/article/pii/S1877050924000103.

The whole methodology was implemented in python language with the use of open-sourced libraries such as tensorflow. Check the requirements.txt to install the employed libraries. 

Initially the CNN scripts (model_timeseries.py, and model_image.py) are used to train and assess a CNN model and to store the predictions (training and testing predictions) from the last dense layer into pickles (as python dictionairies). 

Once the predictions have been stored, the main.py script is used to read the dictionairies, and to perform the next seps of clustering (K-means) and Fuzzy Cognitive Map Construction. 
Finally the decision-making is performed. 

Replace lines in the code custom containing data paths with your own data to use this package. 


