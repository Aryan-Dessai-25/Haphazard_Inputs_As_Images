import os
import argparse
import random
from datetime import datetime
import numpy as np # type: ignore
import torch # type: ignore
import torchvision # type: ignore
import timm # type: ignore
import torch.optim as optim # type: ignore
import torch.nn as nn # type: ignore
import sys
import pickle
from tqdm import tqdm # type: ignore
sys.path.append('/code/DataCode/')
path_to_result = "./Results/"

from Utils import utils, eval_metrics
from DataCode.data_load import dataloader

if __name__ == '__main__':
    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seeding Number')
    parser.add_argument('--plottype', default="bar", type=str,
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

    drop_df, labels, mat_rev_mask, colors=dataloader(data_folder=data_name, drop_rate=drop_rate, seed=seed)
    
    num_inst=drop_df.shape[0]
    num_feats=drop_df.shape[1]
    
    if model_name=='resnet18':
        model=torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model.fc=nn.Linear(model.fc.in_features, 1)
    elif model_name=='resnet34':
        model=torchvision.models.resnet34(weights='IMAGENET1K_V1')
        model.fc=nn.Linear(model.fc.in_features, 1)
    elif model_name=='vit_small_patch16_224':
        model=timm.create_model('vit_small_patch16_224',pretrained=True)
        model.head=nn.Linear(model.head.in_features, 1)
    elif model_name=='vit_base_patch16_224':
        model=timm.create_model('vit_base_patch16_224',pretrained=True)
        model.head=nn.Linear(model.head.in_features, 1)


    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print('timing')
    model=model.to(device) 

    #create lists for storing various data for evaluation and analysis
  
    loss_history=[]
    loss=0.0
    preds=[]
    pred_logits=[]
    true=[]
    acc_history=[]
    f1_history=[]

    
    feat=np.arange(num_feats)
    min_arr=[np.nan]*num_feats
    min_arr=np.array(min_arr)
    max_arr=np.copy(min_arr)
    if plot_type=='bar':
        feat=np.arange(num_feats)

    minmax_time=0
    plot_time=0
    model_time=0
    loss_backprop_time=0
    predict_time=0

    for k in tqdm(range(num_inst)):

        row=drop_df[k]
        rev=mat_rev_mask[k]                         #fetch data to be plotted
        label=labels[k]
        label=torch.tensor(label)
        
        start1=datetime.now()
        norm_row, min_arr, max_arr=utils.minmaxnorm(row,min_arr,max_arr,epsilon=1e-15)    #normalize and update min, max
        end1=datetime.now()
        diff1 = (end1 - start1).total_seconds()
        minmax_time+=diff1

        start2=datetime.now()
        if plot_type=='bar':
            img=utils.bar_tensor(norm_row,rev,colors,feat,dpi=56)        #obtain bar plot tensor 
        elif plot_type=='pie':
            img=utils.pie_tensor(norm_row,colors,dpi=56)
        end2=datetime.now()
        diff2 = (end2 - start2).total_seconds()
        plot_time+=diff2


        img, label = img.to(device), label.to(device)      #transfer to GPU
        img=torch.reshape(img,(-1,3,224,224))      #add extra dimension corresponding to batch, as required by model
        
        optimizer.zero_grad()
        with torch.no_grad():
            start3=datetime.now()
        outputs=model(img)
        with torch.no_grad():
            end3=datetime.now()
            diff3 = (end3 - start3).total_seconds()
            model_time+=diff3
        outputs=torch.reshape(outputs,(-1,))
    
        with torch.no_grad():
            start4=datetime.now() 
        loss=criterion(outputs,label.float())         #compute loss
    
        loss.backward()                               
        loss_history+=[loss.item()]                   #record loss
        
        optimizer.step()
            
    
        with torch.no_grad():

            end4=datetime.now()
            diff4 = (end4 - start4).total_seconds()
            loss_backprop_time+=diff4


            start5=datetime.now()

            predicted=torch.sigmoid(outputs)          # since we use BCEWithLogitsLoss, sigmoid is required for obtaining probability
            predicted=torch.round(predicted)          # get binary prediction
              
            predicted=predicted.to('cpu')
            label=label.to('cpu')
            outputs=outputs.to('cpu')                 #transfer these to cpu to evaluate using sklearn based metrics
            
            
            pred_logits+=[outputs.item()]                    #update lists
            preds.append(predicted.item())
            true.append(label.item())
            end5=datetime.now()
            diff5 = (end5 - start5).total_seconds()
            predict_time+=diff5
    b_acc=eval_metrics.BalancedAccuracy(true,preds)
    auroc=eval_metrics.AUROC(true,pred_logits)
    auprc=eval_metrics.AUPRC(true,pred_logits)
    print(f'Balanced Accuracy= {b_acc}')
    print(f'AUROC score= {auroc}')
    print(f'AUPRC score= {auprc}')
    print('norm=',minmax_time)
    print('plot=', plot_time)
    print('model=', model_time)
    print('bprop=', loss_backprop_time)
    print('predict=', predict_time)
