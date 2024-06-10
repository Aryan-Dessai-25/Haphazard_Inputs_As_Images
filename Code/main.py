import os
import argparse
import random
import numpy as np
import sys
import pickle
sys.path.append('/code/DataCode/')
path_to_result = "./Results/"

from Utils import utils, eval_metrics
from DataCode.data_load import dataloader

if __name__ == '__main__':
    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seeding Number')
    parser.add_argument('--plot_type', default="pie", type=str,
                        choices = ["pie", "bar"], 
                        help='The type of graphical representation.')
    
    # Data Variables
    parser.add_argument('--dataname', default = "magico4", type = str,
                        choices = ["imdb", "higgs", "susy", "a8a", "magic04", 
                                   "spambase", "krvskp", "svmguide3", "ipd", "german", 
                                   "diabetes_f", "wbc", "australian", "wdbc", "ionosphere", "wpbc"],
                        help='The name of the data')
    
    parser.add_argument('--droprate', default = 0.5, type = float, help = "fraction of data unavailable for the model")
    
