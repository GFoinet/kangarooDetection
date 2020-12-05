#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 23:12:28 2020

@author: gilles
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def analyseTraining(train_data : dict, valid_data : dict)->None:
    
    plt.figure(num = 'perfs',figsize=(20,15))
    for i,key in enumerate(train_data.keys()):
        plt.subplot(2,3,i+1)
        plt.plot(train_data[key])
        plt.plot(valid_data[key])
        plt.legend(('train','valid'))
        plt.title(key)
        plt.grid(axis='both')
        
    plt.subplot(4,3,9)
    # sum all losses except 'loss' to verify that loss = sum(all other losses)
    plt.plot(np.sum([np.array(train_data[key]) for key in train_data.keys() if key != 'loss'],axis=0))
    plt.plot(train_data['loss'],'o')
    plt.legend(('cumsum of train loss_*','train loss'))
    plt.grid(axis='both')
    
    plt.subplot(4,3,12)
    # sum all losses except 'loss' to verify that loss = sum(all other losses)
    plt.plot(np.sum([np.array(valid_data[key]) for key in valid_data.keys() if key != 'loss'],axis=0))
    plt.plot(valid_data['loss'],'o')
    plt.legend(('cumsum of valid loss_*','valid loss'))
    plt.grid(axis='both')
    plt.show()
    
def analyseTrainingFromFiles(train_filename : str, valid_filename : str)->None:
    train_data = pickle.load(open(train_filename,'rb'))
    valid_data = pickle.load(open(valid_filename,'rb'))
    analyseTraining(train_data,valid_data)

def analyseTrainingFromPath(path: str)->None:
    (train_data, valid_data)  = pickle.load(open(os.path.join(path,'perfs.p'),'rb'))
    analyseTraining(train_data,valid_data) 

