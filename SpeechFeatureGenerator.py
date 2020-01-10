#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Krishna
"""
import numpy as np
import torch
import feature_extract

class SpeechFeatureGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode='train'):
        """
        Read the textfile and get the paths
        """
        self.audio_links = [line.rstrip('\n') for line in open(manifest)]       

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]

        spec = feature_extract.load_data(audio_link)
        sample = {'spectrogram': torch.from_numpy(spec)}
        return sample
    
    
