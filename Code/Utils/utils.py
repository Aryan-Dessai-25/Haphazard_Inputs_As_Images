import random
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
from torchvision import transforms # type: ignore
import io
import random
from PIL import Image # type: ignore

#--------------Seed--------------#
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

trans=transforms.ToTensor()

#------------Min-Max Normalization----------------------#
#performs min max normalization on supplied data instance with dropped features
#requires running min and max for all features, returnss their updated values
#for the first entry in dataset min max will be intialized to an array consisting of nan values
def minmaxnorm(row,min_arr,max_arr,epsilon=1e-15):
    feats=len(row)                  #obtain number of features
    
    for f in range(feats):           #iterate over the data instance        
        
        if np.isnan(row[f]):        #check if value is dropped, if yes we need not update min and max
            continue                #the normalized row will also have nan for this feature
            
        if np.isnan(min_arr[f]):    # this case is considered when the first entry for a particular feature occurs
            min_arr[f]=row[f]       #in this case we simply assign the value of the first entry to both min and max
            max_arr[f]=row[f]
            continue
            
        if row[f]<min_arr[f]:        #update minimum if applicable
            min_arr[f]=row[f]
        if row[f]>max_arr[f]:        #update maximum if applicable
            max_arr[f]=row[f]
            
    norm_row=(row-min_arr)/(max_arr-min_arr+epsilon)  #min max normalize
    
    return norm_row, min_arr, max_arr

#---------------------Creating Mask For simulating Missing Features---------------------#

#function to create randomized mask matrix with shape identical to that of the dataframe
#takes dataframe dimensions and ratio of dropped features to total features
#returns mask matrix and reverse mask matrix for entire dataset
def create_mask(num_inst,num_feats,drop_rate=0.5,random_seed=42):
    
    dropped_feats=int(num_feats*drop_rate)       #number of features to be dropped
    visible_feats=num_feats- dropped_feats       #number of features visible to the model
    
    mask=np.zeros(dropped_feats)
    m=np.ones(visible_feats)
    mask=np.concatenate((mask,m))       #get array of ones and zeros
    mask=mask.astype(float)                
    mask[mask==0]=[np.nan]             #create a mask with ones and nan values, replacing zeros with nan
    
    mat_mask=np.ones((num_inst,num_feats))       # mask matrix for entire dataset initialized with ones
    mat_rev_mask=np.ones((num_inst,num_feats))   # matrix for plotting crosses at dropped features
    np.random.seed(random_seed)                  #set seed
    
    
    for i in range(num_inst):                    
        
        np.random.shuffle(mask)                  #shuffle mask
        mat_mask[i]*=mask                        #store the mask in matrix
        rev_mask=(np.nan_to_num(mask)-1)*-0.5    #obtain reverse by assigning nan against corresponding 1s in mask
        rev_mask[rev_mask==0]=np.nan             #and 0.5 against correspond nan
        mat_rev_mask[i]*=rev_mask                #store in reverse mask matrix
        
    return mat_mask, mat_rev_mask

#------------------------- Bar Graph Representation-----------------------#

#function to produce image tensors corresponding to given normalized data instance
# takes normalized and dropped values, reverse mask, color scheme and dots per inch resolution
def bar_tensor(values,rev_val,colors,feat,dpi=56):
    s=224/dpi                              #s is the size of image in inches. Models require height and width of image to be 224
    fig, _=plt.subplots(figsize=(s,s))     #fix fig as s*s image. When saving with corresponding dpi we get 224*224 image
    fig=plt.bar(feat,values,color=colors)   #represnt the min-max normed features as bars
    fig=plt.scatter(feat,rev_val,s=180,color='red',marker='x')  #plot crosses at dropped instances
    fig=plt.ylim((0,1))                     #set y axis limits so that all graphs are scaled uniformly
    fig=io.BytesIO()                        #creates byte buffer for converting plt bar object to jpg
    
    figsav=plt.savefig(fig,dpi=dpi, format='jpg') #saving to jpg
    
    image=trans(Image.open(fig))         #opening jpg using PIL and transforming to PyTorch tensor  
    plt.close()
    return image

#-----------------------------Pie Chart Representation---------------------#
def pie_tensor(values, colors, dpi, thresh=0):
    s=224/dpi
    #fig, _=plt.subplots(figsize=(s,s))
    #zero_indices=[i for i, n in enumerate(values) if n <= thresh]
    main_inputs=[]
    main_colors=[]
    lesser_inputs=[]
    lesser_colors=[]
    for i, n in enumerate(values):
        if not pd.isnull(n):
            if n>thresh:
                main_inputs+=[n]
                main_colors+=[colors[i]]
            else:
                lesser_colors+=[colors[i]]
    
    if not lesser_colors:
        lesser_colors=['white']
    if not main_colors:
        main_inputs=[1]
        main_colors=['white']
    
    fig = plt.figure(figsize=(s,s))
    

    #val = np.nan_to_num(values)
    ax1 = fig.add_axes([0, 0, .8, .8], aspect='auto')
    ax1.pie(main_inputs, colors=main_colors, radius = 1.2)

    ax2 = fig.add_axes([.6, .6, .5, .5], aspect='auto')
    one = np.ones_like(lesser_colors)
    ax2.pie(one, colors=lesser_colors, radius = .6)
    
    fig=io.BytesIO()                        #creates byte buffer for converting plt bar object to jpg
    
    figsav=plt.savefig(fig,dpi=dpi, format='jpg') #saving to jpg
    
    image=trans(Image.open(fig))         #opening jpg using PIL and transforming to PyTorch tensor  
    plt.close()
    return image

def bar_cont_tensor(values,colors,feat,dpi=56):
    s=224/dpi
    main_feat=[]
    main_values=[]
    main_colors=[]
    for i, val in enumerate(values):
        if not pd.isnull(val):
            main_feat+=[i]
            main_values+=[val]
            main_colors+=[colors[i]]
    
    if not main_feat:
        main_feat=['none']
        main_values=[0]
        main_colors=['white']
    
    fig, _=plt.subplots(figsize=(s,s))     #fix fig as s*s image. When saving with corresponding dpi we get 224*224 image
    fig=plt.bar(main_feat,main_values,color=main_colors)   #represnt the min-max normed features as bars
    fig=plt.ylim((0,1))                     #set y axis limits so that all graphs are scaled uniformly
    fig=io.BytesIO()                        #creates byte buffer for converting plt bar object to jpg
    
    figsav=plt.savefig(fig,dpi=dpi, format='jpg') #saving to jpg
    
    image=trans(Image.open(fig))         #opening jpg using PIL and transforming to PyTorch tensor  
    image=transforms.functional.rotate(image,angle=270)
    plt.close()
    return image