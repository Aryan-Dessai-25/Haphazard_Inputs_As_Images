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

    colors=[(142, 102, 82),
    (13, 13, 20),
    (47, 143, 161),
    (117, 98, 107),
    (143, 120, 133),
    (56, 150, 142),
    (193, 178, 183),
    (219, 145, 180),
    (180, 135, 27),
    (106, 117, 108),
    (13, 78, 78),
    (207, 77, 46),
    (4, 7, 7),
    (159, 128, 59),
    (157, 166, 176),
    (222, 184, 202),
    (122, 192, 93),
    (88, 99, 97),
    (9, 26, 21),
    (52, 14, 10),
    (20, 17, 20),
    (209, 176, 115),
    (255, 187, 151),
    (231, 215, 233),
    (99, 91, 24),
    (66, 50, 32),
    (11, 50, 66),
    (95, 182, 188),
    (1, 29, 31),
    (20, 158, 168),
    (190, 199, 255),
    (44, 36, 19),
    (149, 27, 78),
    (209, 161, 107),
    (37, 16, 77),
    (211, 214, 205),
    (66, 48, 50),
    (222, 177, 202),
    (46, 50, 37),
    (5, 6, 11),
    (239, 242, 244),
    (163, 65, 149),
    (107, 124, 83),
    (111, 96, 99),
    (79, 150, 95),
    (242, 217, 203),
    (34, 20, 44),
    (129, 137, 148),
    (98, 48, 46),
    (202, 186, 173),
    (22, 139, 85),
    (239, 244, 199),
    (33, 40, 45),
    (94, 117, 74),
    (207, 206, 203),
    (55, 218, 167),
    (158, 133, 123),
    (188, 196, 181),
    (147, 148, 152),
    (55, 31, 90),
    (154, 114, 58),
    (10, 7, 12),
    (4, 63, 72),
    (145, 98, 81),
    (86, 35, 4),
    (195, 106, 35),
    (13, 50, 63),
    (25, 116, 92),
    (221, 223, 218),
    (13, 33, 62),
    (169, 150, 171),
    (223, 213, 254),
    (5, 17, 20),
    (176, 182, 181),
    (1, 1, 1),
    (108, 168, 218),
    (12, 101, 161),
    (66, 80, 84),
    (45, 60, 31),
    (205, 204, 179),
    (197, 175, 225),
    (178, 197, 199),
    (33, 16, 5),
    (186, 172, 134),
    (250, 245, 247),
    (142, 171, 28),
    (8, 5, 10),
    (248, 246, 254),
    (37, 40, 29),
    (12, 134, 132),
    (50, 12, 51),
    (112, 176, 200),
    (97, 156, 188),
    (120, 102, 112),
    (96, 51, 21),
    (105, 56, 42),
    (79, 86, 88),
    (32, 42, 8),
    (248, 246, 252),
    (165, 109, 43),
    (196, 150, 130),
    (173, 101, 170),
    (200, 201, 207),
    (121, 115, 124),
    (28, 15, 15),
    (132, 132, 75),
    (42, 79, 102),
    (77, 85, 87),
    (206, 208, 175),
    (43, 13, 48),
    (35, 40, 39),
    (114, 107, 88),
    (65, 65, 64),
    (236, 238, 237),
    (91, 92, 43),
    (25, 88, 98),
    (195, 173, 201),
    (250, 249, 250),
    (141, 158, 184),
    (168, 121, 132),
    (55, 19, 11),
    (58, 54, 39),
    (204, 221, 241)]
    return X, Y, colors

