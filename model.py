#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:51:09 2020

@author: krishna.dn
"""

from torch import nn
import torch
class CAE(nn.Module):
    '''
    Convlution autoencoder model takes 256x200 dimensional matrix as input
    and projects it into 256 dimensional vector representation
    '''
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                ### encoder layer 1
                nn.Conv2d(1, 64,8 , stride=3, padding=1),
                nn.ReLU(True),
                ### Encoder layer 2
                nn.Conv2d(64, 64, 5, stride=1, padding=1),
                nn.ReLU(True),
                ## Encoder layer 3
                nn.Conv2d(64, 128, 5, stride=1, padding=1),
                nn.ReLU(True),
                ### Encoder layer 4
                nn.Conv2d(128, 128, 5, stride=1, padding=1),
                nn.ReLU(True),
                ### Encoder layer 5
                nn.Conv2d(128, 128, 5, stride=1, padding=1),
                nn.ReLU(True),
                ### Encoder layer 6
                nn.Conv2d(128, 128, 5, stride=2, padding=1),
                nn.ReLU(True),
                ### Encoder layer 7
                nn.Conv2d(128, 32, 5, stride=2, padding=1),
                nn.ReLU(True))
        self.enc_bottleneck = nn.Linear(7488,256)
        self.dec_bottleneck = nn.Linear(256,7488)
        ######## Decoder
        self.decoder = nn.Sequential(
                ## Decoder layer 1
                nn.ConvTranspose2d(32,128,5, stride=2,padding=1),
                nn.ReLU(True),
                ## Decoder layer 2
                nn.ConvTranspose2d(128,128,5, stride=2,padding=1),
                nn.ReLU(True),
                ## Decoder layer 3
                nn.ConvTranspose2d(128,128,5, stride=1,padding=1),
                nn.ReLU(True),
                ## Decoder layer 4
                nn.ConvTranspose2d(128,128,5, stride=1,padding=1),
                nn.ReLU(True),
                ## Decoder layer 5
                nn.ConvTranspose2d(128,64,5, stride=1,padding=1),
                nn.ReLU(True),
                ## Decoder layer 6
                nn.ConvTranspose2d(64,64,5, stride=1,padding=1),
                nn.ReLU(True),
                ## Decoder layer 7
                nn.ConvTranspose2d(64,1,(10,14), stride=3,padding=0),
                nn.ReLU(True),
                )

    def forward(self, x):
        
        encoder_out = self.encoder(x)
        reshape_enc_out = torch.flatten(encoder_out,start_dim=1)
        enc_bnf = self.enc_bottleneck(reshape_enc_out)
        dec_bnf = self.dec_bottleneck(enc_bnf)
        reshape_dec_out = dec_bnf.reshape([encoder_out.shape[0],encoder_out.shape[1],encoder_out.shape[2],encoder_out.shape[3]])
        dec_out = self.decoder(reshape_dec_out)
        return enc_bnf,dec_out
    
    
    
    