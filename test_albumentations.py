#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:17:23 2020

@author: gilles
"""
import albumentations as A
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import xml.etree.ElementTree as ET
import os
import torch
import utils
import matplotlib.pyplot as plt
from matplotlib import patches
from visualize_bbox import visualize

idx_to_label = {
    0: 'background',
    1: 'kangaroo',
    2: 'other'
}
label_to_idx = {v: k for k, v in idx_to_label.items()}

def get_bboxes(annot_path):
    tree = ET.parse(annot_path)
    root = tree.getroot()
    target_list = []
    bboxes_list = []
    for tag in root.iter('object'):
        target_list.append(label_to_idx[tag.find('name').text])
        xmin = int(tag.find('bndbox/xmin').text)
        xmax = int(tag.find('bndbox/xmax').text)
        ymin = int(tag.find('bndbox/ymin').text)
        ymax = int(tag.find('bndbox/ymax').text)
        bboxes_list.append([xmin, ymin, xmax, ymax])
    return target_list, bboxes_list

class kangarooDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.imgs = self.imgs[0:1]
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annots", self.imgs[idx][:-3]+'xml')
        img = Image.open(img_path).convert("RGB")
        # get bounding box coordinates for each mask
        targets, boxes = get_bboxes(annot_path)
        boxes_ = boxes.copy()
        num_objs = len(targets)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(targets, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # # img.show()
        # fig = plt.figure(1)
        # #add axes to the image
        # ax = fig.add_axes([0,0,1,1])

        # print(np.shape(np.array(img)))
        # plt.imshow(np.array(img))
        # for i in range(num_objs):
        #     rect = patches.Rectangle((boxes[i,0],boxes[i,1]), boxes[i,2]-boxes[i,0], boxes[i,3]-boxes[i,1], edgecolor = 'r', facecolor = 'none')
        #     ax.add_patch(rect)
        # plt.show()
        # visualize(img,img_out,targets,idx_to_label)
        
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(img),bboxes=boxes_,class_labels=targets)
        img_out = transformed['image']
        bboxes_out = transformed['bboxes']
        targets_out = transformed['class_labels']
        # fig = plt.figure(2)
        # ax = fig.add_axes([0,0,1,1])
        # plt.imshow(img_out)
        # print(type(bboxes_out))
        # print(len(bboxes_out))
        # print(bboxes_out[0])
        # print(bboxes_out[1])
        # for i in range(num_objs):
        #     rect = patches.Rectangle((bboxes_out[i][0],bboxes_out[i][1]), bboxes_out[i][2]-bboxes_out[i][0], bboxes_out[i][3]-bboxes_out[i][1], edgecolor = 'r', facecolor = 'none')
        #     ax.add_patch(rect)
        # plt.show()
        visualize(img_out,bboxes_out,targets_out,idx_to_label)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
    
def get_transform(train):
    transforms = []
    transforms.append(A.RandomBrightnessContrast(p=1))
    transforms.append(A.RandomGamma(p=1))
    transforms.append(A.CLAHE(p=1))
    transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.ShiftScaleRotate(rotate_limit=15,p=0.5))
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


dataset = kangarooDataset('kangaroo', get_transform(train=True))

data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

for images, targets in data_loader:
    images = list(img for img in images)
    