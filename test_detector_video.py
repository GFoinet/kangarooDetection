#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:53:26 2020

@author: sirehna
"""
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
import glob
import cv2
from visualize_bbox import visualize
import numpy as np

idx_to_label = {
    0: 'background',
    1: 'kangaroo',
    2: 'other'
}
label_to_idx = {v: k for k, v in idx_to_label.items()}

cap = cv2.VideoCapture('kangaroos.mp4')
fps_in = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videowriter_out = cv2.VideoWriter('output.avi',fourcc,int(fps_in),(width,height))
device = torch.device('cuda')
model = torch.load('/home/gilles/ML/kangarooDetection/experiments/20201207_18h12:53s/test.pt')
model.to(device)
model.eval()

transformer = T.Compose([
    T.ToTensor(),
    ])

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (ret == True):
        img = np.array(Image.fromarray(frame).convert("RGB"))
        imgt = transformer(img).unsqueeze(0)
        pred = model(imgt.to(device))
        img_out = visualize(img,pred[0]['boxes'].tolist(),pred[0]['labels'].tolist(),idx_to_label,show=False)
        videowriter_out.write(img_out)
    else:
        cap.release()
videowriter_out.release()
