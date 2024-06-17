import numpy as np
import pandas as pd
import os
import pickle
from Utils.utils import seed_everything

def data_folder_path(data_folder, data_name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', data_folder, data_name)

def data_load_magico4(data_folder,seed):
    data_name = "magico4.csv"
    data_path = data_folder_path(data_folder, data_name)
    
    seed_everything(seed)
    df =  pd.read_csv(data_path).sample(frac=1).reset_index(drop=True)
    labels=df['class'].to_numpy()
    labels[labels=='h']=[0]
    labels[labels=='g']=[1]
    labels=labels.astype(float)          #we extract labels and assign floting point binary labels
    
    df=df.drop(['class','Unnamed: 0'],axis=1) #drop the labels and indexing from main dataframe
    df=df.to_numpy()
    
    return df, labels

def data_load_a8a(data_folder):
    data_name = "a8a.txt"
    n_feat = 123
    number_of_instances = 32561
    data_path = data_folder_path(data_folder, data_name)
    data_initial =  pd.read_csv(data_path, sep = " ", header = None, engine = 'python')
    data = pd.DataFrame(0, index=range(number_of_instances), columns = list(range(1, n_feat+1)))
    # 16th column contains only NaN value
    data_initial = data_initial.iloc[:, :15]
    for j in range(data_initial.shape[0]):
            l = [int(i.split(":")[0])-1 for i in list(data_initial.iloc[j, 1:]) if not pd.isnull(i)]
            data.iloc[j, l] = 1
    label = np.array(data_initial[0] == -1)*1
    data.insert(0, column='class', value=label)
    data = data.sample(frac = 1)

    Y = np.array(data.iloc[:,:1])
    X = np.array(data.iloc[:,1:])

    return X, Y

