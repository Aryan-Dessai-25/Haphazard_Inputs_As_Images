import numpy as np
from Utils.utils import minmaxnorm, create_mask, bar_tensor, pie_tensor
from DataCode.data_utils import data_load_magico4

def dataloader(data_folder='magico4',drop_rate=0.5,seed=42):
    
    X,Y=data_load_magico4(data_folder)
    num_inst=X.shape[0]
    num_feats=X.shape[1]
    mat_mask, mat_rev_mask=create_mask(num_inst,num_feats,drop_rate=0.5,seed)
    X_haphazard=np.multiply(X,mat_mask)

return X_haphazard, Y, mat_rev_mask
        
        
