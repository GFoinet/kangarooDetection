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

cap = cv2.VideoCapture('kangaroos_640_512.avi')
device = torch.device('cuda')
model = torch.load('test.pt')
model.to(device)
model.eval()

transformer = T.Compose([
    T.ToTensor(),
    ])

imgList = glob.glob('test/*.jpg')

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (ret == True):
        img = Image.fromarray(frame).convert("RGB")

        imgt = transformer(img).unsqueeze(0)
        pred = model(imgt.to(device))
        print(pred)
        
        # draw
        base = img.convert("RGBA")
        txt = Image.new("RGBA", base.size, (255,255,255,0))
        # get a drawing context
        d = ImageDraw.Draw(txt)
        for bbox, score in zip(pred[0]['boxes'].tolist(),pred[0]['scores'].tolist()):
            if score > 0.:
                d.rectangle(bbox,fill=None,width = 10)
        
        out = Image.alpha_composite(base, txt)
        if(len(bbox)>=1):
            out.show()
