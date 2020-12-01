#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:47:01 2020

@author: sirehna
"""

import torch

model = torch.load('test.pt')
model.eval()
model.to('cpu')
batch_size = 1
x = torch.randn(3, 1080, 1920, requires_grad=False)

torch_out = model([x, x])
torch.onnx.export(model,               # model being run
                  [x, x],                         # model input (or a tuple for multiple inputs)
                  "kangaroo_detector_1080_1920_.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
