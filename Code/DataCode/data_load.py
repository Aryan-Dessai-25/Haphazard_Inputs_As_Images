import numpy as np
from Utils.utils import 
from DataCode.data_utils import data_load_magico4

def dataloader(data_folder='magico4',plot_type='pie',drop_rate=0.5):
    
    X,Y=data_load_magico4(data_folder)
    if(plot_type=='bar'):
        
