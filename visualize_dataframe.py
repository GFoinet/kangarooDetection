#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:13:09 2020

@author: gilles
"""



# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import os


kangaroo_path = '/home/gilles/ML/fasterRCNN/kangaroo'
kangaroo_imgs_path = os.path.join(kangaroo_path, 'images')
kangaroo_annots_path = os.path.join(kangaroo_path, 'annots')

# read the csv file using read_csv function of pandas
kangaroo_dataframe = pd.read_csv('kangaroo_dataframe.csv')
train = kangaroo_dataframe.iloc[:200]
test = kangaroo_dataframe.iloc[201:]
train.head()

# reading single image using imread function of matplotlib
image = plt.imread(os.path.join(kangaroo_imgs_path,train.iloc[0].img_names))
plt.imshow(image)

# Number of unique training images
print('images uniques = {}'.format(train['img_names'].nunique()))

fig = plt.figure()

#add axes to the image
ax = fig.add_axes([0,0,1,1])

# read and plot the image
img_to_plot = train.iloc[10].img_names
image = plt.imread(os.path.join(kangaroo_imgs_path,img_to_plot))
plt.imshow(image)

# iterating over the image for different objects
for _,row in train[train.img_names == img_to_plot].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    
    width = xmax - xmin
    height = ymax - ymin
    
    # assign different color to different classes of objects
    if row.animal_type == 'kangaroo':
        edgecolor = 'r'
        ax.annotate('kangaroo', xy=(xmax-40,ymin+20))
    else:
        edgecolor = 'g'
        ax.annotate('unknown', xy=(xmax-40,ymin+20))
        
    # add bounding boxes to the image
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
    
    ax.add_patch(rect)

# create .txt file
data = pd.DataFrame()
data['format'] = train['img_names']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['animal_type'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')