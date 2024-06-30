import numpy as np # type: ignore
from Utils.utils import minmaxnorm, create_mask, bar_tensor, pie_tensor
from DataCode.data_utils import data_load_magico4, data_load_a8a, data_load_susy

def dataloader(data_folder='magico4',drop_rate=0.5,seed=42):

    if data_folder=='magico4':
        X,Y,colors,rgbcol=data_load_magico4(data_folder,seed)
    elif data_folder=='a8a':
        X,Y,colors=data_load_a8a(data_folder)
    elif data_folder=='SUSY':
        X,Y,colors,rgbcol=data_load_susy(data_folder)
    num_inst=X.shape[0]
    num_feats=X.shape[1]
    mat_mask, mat_rev_mask=create_mask(num_inst,num_feats,drop_rate,random_seed=seed)
    X_haphazard=np.multiply(X,mat_mask)

    return X_haphazard, Y, mat_rev_mask, colors, rgbcol
        
        
