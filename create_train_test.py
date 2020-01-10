#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:46:16 2020

@author: krishna.dn
"""

import os
import glob
import numpy as np
import random


root_dir = 'audioset/'
all_files = sorted(glob.glob(root_dir+'/*.wav'))
test_ind = random.sample(range(len(all_files)),int(0.2*len(all_files)))
train_fid = open('meta/traning.txt','w')
test_fid = open('meta/testing.txt','w')

for i in test_ind:
    filepath = all_files[i]
    test_fid.write(filepath+'\n')
test_fid.close()


for i in range(len(all_files)):
    if i in test_ind:
        continue
    else:
        filepath = all_files[i]
        train_fid.write(filepath+'\n')
train_fid.close()
