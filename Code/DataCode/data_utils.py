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
    labels=labels.reshape(-1,1)
    df=df.drop(['class','Unnamed: 0'],axis=1) #drop the labels and indexing from main dataframe
    df=df.to_numpy()
    colors=['sandybrown','blue','black','magenta','olive','red','green','slategray','turquoise','yellow']
    return df, labels, colors

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

    colors=['#8e6652',
    '#0d0d14',
    '#2f8fa1',
    '#75626b',
    '#8f7885',
    '#38968e',
    '#c1b2b7',
    '#db91b4',
    '#b4871b',
    '#6a756c',
    '#0d4e4e',
    '#cf4d2e',
    '#040707',
    '#9f803b',
    '#9da6b0',
    '#deb8ca',
    '#7ac05d',
    '#586361',
    '#091a15',
    '#340e0a',
    '#141114',
    '#d1b073',
    '#ffbb97',
    '#e7d7e9',
    '#635b18',
    '#423220',
    '#0b3242',
    '#5fb6bc',
    '#011d1f',
    '#149ea8',
    '#bec7ff',
    '#2c2413',
    '#951b4e',
    '#d1a16b',
    '#25104d',
    '#d3d6cd',
    '#423032',
    '#deb1ca',
    '#2e3225',
    '#05060b',
    '#eff2f4',
    '#a34195',
    '#6b7c53',
    '#6f6063',
    '#4f965f',
    '#f2d9cb',
    '#22142c',
    '#818994',
    '#62302e',
    '#cabaad',
    '#168b55',
    '#eff4c7',
    '#21282d',
    '#5e754a',
    '#cfcecb',
    '#37daa7',
    '#9e857b',
    '#bcc4b5',
    '#939498',
    '#371f5a',
    '#9a723a',
    '#0a070c',
    '#043f48',
    '#916251',
    '#562304',
    '#c36a23',
    '#0d323f',
    '#19745c',
    '#dddfda',
    '#0d213e',
    '#a996ab',
    '#dfd5fe',
    '#051114',
    '#b0b6b5',
    '#010101',
    '#6ca8da',
    '#0c65a1',
    '#425054',
    '#2d3c1f',
    '#cdccb3',
    '#c5afe1',
    '#b2c5c7',
    '#211005',
    '#baac86',
    '#faf5f7',
    '#8eab1c',
    '#08050a',
    '#f8f6fe',
    '#25281d',
    '#0c8684',
    '#320c33',
    '#70b0c8',
    '#619cbc',
    '#786670',
    '#603315',
    '#69382a',
    '#4f5658',
    '#202a08',
    '#f8f6fc',
    '#a56d2b',
    '#c49682',
    '#ad65aa',
    '#c8c9cf',
    '#79737c',
    '#1c0f0f',
    '#84844b',
    '#2a4f66',
    '#4d5557',
    '#ced0af',
    '#2b0d30',
    '#232827',
    '#726b58',
    '#414140',
    '#eceeed',
    '#5b5c2b',
    '#195862',
    '#c3adc9',
    '#faf9fa',
    '#8d9eb8',
    '#a87984',
    '#37130b',
    '#3a3627',
    '#ccddf1']
    return X, Y, colors

