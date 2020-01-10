#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:22:26 2019

@author: Krishna
"""



from torch.utils.data import DataLoader   
import torch
from SpeechFeatureGenerator import SpeechFeatureGenerator
import torch.nn as nn
import os
from torch import optim
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.multiprocessing
import numpy as np
from model import CAE
from feature_extract import speech_collate
torch.multiprocessing.set_sharing_strategy('file_system')


## Hyperparams
num_epoch = 50

#### Dataset info
data_path_training = 'meta/traning.txt'
data_path_testing = 'meta/testing.txt'


### Data related
dataset_train_classify = SpeechFeatureGenerator(manifest=data_path_training)
dataloader_train_classify = DataLoader(dataset_train_classify, batch_size=10, shuffle=True,collate_fn=speech_collate) 

dataset_test_classify = SpeechFeatureGenerator(manifest=data_path_testing)
dataloader_test_classify = DataLoader(dataset_test_classify, batch_size=10, shuffle=True,collate_fn=speech_collate) 

### Cuda specific
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

####### Model specification
model = CAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_func = nn.MSELoss()


for epoch in range(num_epoch):
    total_loss = []
    for (i,data) in enumerate(dataloader_train_classify):
        features = torch.from_numpy(np.asarray([torch_tensor.numpy().reshape(1,256,200) for torch_tensor in data]))
        features = features.to(device)
        bnf,rec_x = model(features)
        loss = loss_func(rec_x,features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        total_loss.append(loss.item())
    mean_loss = np.mean(np.asarray(total_loss))
    print('Mean loss {}  after {} epochs'.format(mean_loss,epoch))


        
    
model_save_path = os.path.join('trained_models', 'check_point_'+str(epoch))
state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
torch.save(state_dict, model_save_path)



'''
    ### Evaluate the model
    model.eval()
    total_acc_lid = []
    total_acc_gid = []
    
    for i_batch, sample_batched in enumerate(dataloader_test):
        if i_batch==500:
            break
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
        labels_lid = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
        labels_gender = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])) 
    
        features, labels_lid,labels_gid = features.to(device), labels_lid.to(device), labels_gender.to(device)
        
        lid_preds,gender_preds = model(features)
        prediction_lid = np.argmax(lid_preds.detach().cpu().numpy(),axis=1)
        prediction_gid = np.argmax(gender_preds.detach().cpu().numpy(),axis=1)
        accuracy_lid = accuracy_score(labels_lid.detach().cpu().numpy(),prediction_lid)
        accuracy_gid = accuracy_score(labels_gid.detach().cpu().numpy(),prediction_gid)
        total_acc_lid.append(accuracy_lid)
        total_acc_gid.append(accuracy_gid)
        
        
    final_test_acc = np.sum(np.asarray(total_acc_lid))/len(total_acc_lid)
    print('The test accuracy {} for epoch {}'.format(final_test_acc,epoch))
'''
