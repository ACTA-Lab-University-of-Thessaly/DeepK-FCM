import numpy as np 
import pandas as pd 



def convert_dic_to_categorical_df(experts):
    classes = experts.keys()
    lis = []
    labels = []
    for clas in classes:
        for exp in experts[clas]:
            lis.append(experts[clas][exp])
            labels.append(clas)

    lis = np.array(lis)
    zeros = np.zeros(len(labels))
    columns = []
    total_centroids = len(lis[0])
    name = 'centroid_'
    for i in classes:
        for j in range(total_centroids//len(classes)):
            columns.append(name + str(i) + '_' + str(j))

    df = pd.DataFrame(lis, columns = columns)
    for clas in classes:
        df[f'output_{clas}'] = zeros
    
    for i in range(len(labels)):
        df[f'output_{labels[i]}'].iloc[i] = 1
    return df

