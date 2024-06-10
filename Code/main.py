import os
import argparse
import random
import numpy as np
import torch
import torchvision
import timm
import torch.optim as optim
import torch.nn as nn
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
    parser.add_argument('--plottype', default="pie", type=str,
                        choices = ["pie", "bar"], 
                        help='The type of graphical representation.')
    
    # Data Variables
    parser.add_argument('--dataname', default = "magico4", type = str,
                        choices = ["imdb", "higgs", "susy", "a8a", "magic04"],
                        help='The name of the data')
    
    parser.add_argument('--droprate', default = 0.5, type = float, help = "fraction of data unavailable for the model")
    parser.add_argument('--modelname', default = "resnet34", type = str,
                        choices = ["resnet18", "resnet34", "vit_small_patch16_224", "vit_base_patch16_224"],
                        help = "The name of the model used")
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    args = parser.parse_args()
    seed = args.seed
    plot_type = args.plottype
    data_name = args.dataname
    drop_rate = args.droprate
    model_name=args.modelname
    lr=args.lr
    utils.seed_everything(seed)

    drop_df, labels, mat_rev_mask=dataloader(data_folder=data_name, drop_rate=drop_rate, seed=seed)
    
    num_inst=drp_df.shape[0]
    num_feats.shape[1]
    
    if model_name=='resnet18':
        model=torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model.fc=nn.Linear(model.fc.in_features, 1)
    elif model_name=='resnet34':
        model=torchvision.models.resnet34(weights='IMAGENET1K_V1')
        model.fc=nn.Linear(model.fc.in_features, 1)
    elif model_name='vit_small_patch16_224':
        model=timm.create_model('vit_small_patch16_224',pretrained=True)
        model.head=nn.Linear(model.head.in_features, 1)
    elif model_name='vit_base_patch16_224':
        model=timm.create_model('vit_base_patch16_224',pretrained=True)
        model.head=nn.Linear(model.head.in_features, 1)


    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(),lr=cfg.lr)
    device='cuda' if torch.cuda.is_available() else 'cpu
    model=model.to(device) 

    #create lists for storing various data for evaluation and analysis
  
    loss_history=[]
    loss=0.0
    preds=[]
    pred_logits=[]
    true=[]
    acc_history=[]
    f1_history=[]
    
    colors=['sandybrown','blue','black','magenta','olive','red','green','slategray','turquoise','yellow']
    feat=np.arange(num_feats)
    min_arr=[np.nan]*num_feats
    min_arr=np.array(min_arr)
    max_arr=np.copy(min_arr)

    for k in tqdm(range(num_inst)):

        row=drop_df[k]
        rev=mat_rev_mask[k]                         #fetch data to be plotted
        label=labels[k]
        label=torch.tensor([label])
        
        norm_row, min_arr, max_arr=minmaxnorm(row,min_arr,max_arr,epsilon=1e-15)    #normalize and update min, max
        img=bar_tensor(norm_row,rev,colors,dpi=56)        #obtain bar plot tensor 
    
        
        img, label = img.to(device), label.to(device)      #transfer to GPU
        img=torch.reshape(img,(-1,3,224,224))      #add extra dimension corresponding to batch, as required by model
        
        optimizer.zero_grad()
        outputs=model(img)
        outputs=torch.reshape(outputs,(-1,))
    
            
        loss=criterion(outputs,label.float())         #compute loss
    
        loss.backward()                               
        loss_history+=[loss.item()]                   #record loss
        
        optimizer.step()
            
    
        with torch.no_grad():
                
            predicted=torch.sigmoid(outputs)          # since we use BCEWithLogitsLoss, sigmoid is required for obtaining probability
            predicted=torch.round(predicted)          # get binary prediction
                
            predicted=predicted.to('cpu')
            label=label.to('cpu')
            outputs=outputs.to('cpu')                 #transfer these to cpu to evaluate using sklearn based metrics
            
            
            pred_logits+=[outputs]                    #update lists
            preds.append(predicted)
            true.append(label)
    b_acc=BalancedAccuracy(true,preds)
    auroc=AUROC(true,pred_logits)
    auprc=AUPRC(true,pred_logits)
    print(f'Balanced Accuracy= {b_acc}')
    print(f'AUROC score= {auroc}')
    print(f'AUPRC score= {auprc}')
