from __future__ import division
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image
from model.yolo import YoloBody
import matplotlib.pyplot as plt
import matplotlib.patches as patches
anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
def get_classes(classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

class YOLOpredict(object):
    def __init__(self,filepath,model_path,class_path,input_shape = [416,416]) -> None:
        class_names,num_classes = get_classes(class_path)
        self.num_classes = num_classes
        self.class_names = class_names
        self.model = YoloBody(anchors_mask=anchors_mask,num_classes=num_classes).cuda()
        state_dict = torch.load(model_path, map_location="cuda")
        for i,v in  state_dict.items():
            k = i  
            break
        if(k[:7]=="module."):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.filepath = filepath
        self.input_shape = input_shape
        self.threshold= 0.1
    def decode_box(self,outputs):
        pred_box = []
        for l,output in enumerate(outputs):
            #output是1，75，13，13
            batchsize = output.size(0)
            in_h = output.size(2)
            in_w = output.size(3)
            stride_h = self.input_shape[0]/in_h
            stride_w = self.input_shape[1]/in_w
            #输入数据的格式是batchsize，3*（5+num_classes),13,13/26,26/52,52
            prediction = output.view(batchsize,3,5+self.num_classes,in_h,in_w).permute(0,1,3,4,2).contiguous()
            #框的中心位置调整参数
            x = torch.sigmoid(prediction[...,0])
            y = torch.sigmoid(prediction[...,1])
            #框的宽高调整参数
            w = prediction[...,2]
            h = prediction[...,3]
            #框内有没有物体的置信度
            conf = torch.sigmoid(prediction[...,4])
            #种类置信度
            cls = torch.sigmoid(prediction[...,5:])
            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
                batchsize * len(self.anchors_mask[l]), 1, 1).view(x.shape).type(torch.cuda.FloatTensor)
            grid_y = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
                batchsize * len(self.anchors_mask[l]), 1, 1).view(y.shape).type(torch.cuda.FloatTensor)
            FloatTensor = torch.cuda.FloatTensor
            LongTensor  = torch.cuda.LongTensor
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors[anchors_mask[l]]]

            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batchsize, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(batchsize, 1).repeat(1, 1, in_h * in_w).view(h.shape)
            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x.data + grid_x
            pred_boxes[..., 1]  = y.data + grid_y
            pred_boxes[..., 2]  = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3]  = torch.exp(h.data) * anchor_h
            _scale = torch.Tensor([in_w, in_h, in_w, in_h]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batchsize, -1, 4) / _scale,
                                conf.view(batchsize, -1, 1), cls.view(batchsize, -1, self.num_classes)), -1)
            pred_box.append(output.data)
        return pred_box
    def predict_image(self):
        image = Image.open(self.filepath)
        image = image.convert("RGB").resize(self.input_shape)
        #1,3,416,416
        image  = np.expand_dims(np.transpose(np.array(image, dtype='float32'), (2, 0, 1)), 0)
        image = torch.tensor(image).cuda()
        outputs = self.model(image)
        outputs = self.decode_box(outputs)

        
        


            


if __name__ == "__main__":
    filepath = "000089.jpg"
    model_path = "model_data\yolo_weights.pth"
    class_path = "model_data\coco_classes.txt"
    predict = YOLOpredict(filepath,model_path,class_path)
    predict.predict_image()