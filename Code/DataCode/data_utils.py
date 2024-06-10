import numpy as np
import pandas as pd
import os
import pickle
from Utils.utils import seed_everything

def data_folder_path(data_folder, data_name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data', data_folder, data_name)

def data_load_magic04(data_folder,seed):
    data_name = "magic04.csv"
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

