# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import xml.etree.ElementTree as ET
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pickle
from datetime import datetime
from analyse_training import analyseTrainingFromPath

CFG = {
       'training_batch_size':2,
       'training_EPOCHS':30,
       'valid_batch_size':1,
       'SGD_lr':0.01,
       'SGD_momentum':0.9,
       'SGD_weights_decay':0.0,
       'LRscheduler_step_size':10,
       'LRscheduler_gamma':1
       }
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

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annots", self.imgs[idx][:-3]+'xml')
        img = Image.open(img_path).convert("RGB")
        # get bounding box coordinates for each mask
        targets, boxes = get_bboxes(annot_path)
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)




def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=5)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from engine import train_one_epoch, evaluate_GFO
import utils


def main(exp_folder:str='')->None:
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = kangarooDataset('kangaroo', get_transform(train=True))
    dataset_test = kangarooDataset('kangaroo', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    indice_split = 20*len(dataset)//100
    dataset = torch.utils.data.Subset(dataset, indices[:-indice_split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-indice_split:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=CFG['training_batch_size'], shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=CFG['valid_batch_size'], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=CFG['SGD_lr'],
                                momentum=CFG['SGD_momentum'], weight_decay=CFG['SGD_weights_decay'])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=CFG['LRscheduler_step_size'],
                                                   gamma=CFG['LRscheduler_gamma'])

    # let's train it

    train_losses_dict = {'loss': [], 'loss_classifier': [], 'loss_box_reg': [], 'loss_objectness': [], 'loss_rpn_box_reg': []}
    eval_losses_dict = {'loss': [], 'loss_classifier': [], 'loss_box_reg': [], 'loss_objectness': [], 'loss_rpn_box_reg': []}
    
    for epoch in range(CFG['training_EPOCHS']):
        # train for one epoch, printing every 10 iterations
        train_metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        for k in train_losses_dict.keys():
            train_losses_dict[k].append(train_metrics.meters[k].avg)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        eval_metrics = evaluate_GFO(model, data_loader_test, device=device, print_freq=10)
        for k in eval_losses_dict.keys():
            eval_losses_dict[k].append(eval_metrics.meters[k].avg)
    
    torch.save(model,os.path.join(exp_folder,'test.pt'))
    pickle.dump([train_losses_dict, eval_losses_dict],open(os.path.join(exp_folder,'perfs.p'),'wb'))
    pickle.dump(CFG,open(os.path.join(exp_folder,'config.p'),'wb'))
    print("That's it!")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    now = datetime.now()
    experiment_folder_name = 'experiments/'
    if not os.path.exists(experiment_folder_name):
        os.makedirs(experiment_folder_name)
    experiment_name = now.strftime("%d%Y%m%d_%Hh%m:%Ss")
    os.mkdir(os.path.join(experiment_folder_name,experiment_name))
    exp_path = os.path.join(experiment_folder_name,experiment_name)
    main(exp_path)
    analyseTrainingFromPath(exp_path)
    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
