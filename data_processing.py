#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:07:12 2020

@author: krishna.dn
"""

import os
import numpy as np
import glob
from shutil import copyfile
from shutil import rmtree



unbalenced_file = 'data_files/unbalanced_train_segments.csv'
balenced_file = 'data_files/balanced_train_segments.csv'

read_ub_file = [line.rstrip('\n') for line in open(balenced_file)]

youtube_base = 'https://www.youtube.com/watch?v='
for i in range(3,len(read_ub_file)):
    line = read_ub_file[i]
    yt_id = line.split(',')[0]
    final_id =youtube_base+yt_id
    try:
        if not os.path.exists('youtube_vid/'):
            os.makedirs('youtube_vid/')
        command_download = 'youtube-dl -f 18 '+final_id+' -o youtube_vid/temp.mp4'
        os.system(command_download)
    except:
        print('Download Error')
    
    vid_filepaths = sorted(glob.glob('youtube_vid/*'))
    if vid_filepaths:
        filepath = vid_filepaths[0]
        extract_audio = 'ffmpeg -i '+filepath+' -f wav youtube_vid/temp.wav'
        os.system(extract_audio)
        downsample = 'sox youtube_vid/temp.wav -r 16k -c 1 youtube_vid/temp_16k.wav'
        os.system(downsample)
        src_path = 'youtube_vid/temp_16k.wav'
        dest_path = 'audioset/'+yt_id+'.wav'
        copyfile(src_path,dest_path)
        rmtree('youtube_vid/')
        
#################################################################################

        
        
