import datetime
import os
from random import shuffle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import nn
from model.yolo import YoloBody
from utils.loss import YOLOLoss
from utils.dataloader import YOLOdataloader,yolo_dataset_collate
from utils.utils import get_lr_scheduler,set_optimizer_lr
anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
num_anchors = 9
if __name__ == "__main__":
    Cuda = True
    classes_path = "voc_classes.txt"
    anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
    model_path = "save\ep300-loss0.022.pth"
    input_shape = [416,416]
    pretrained = False
    save_period = 5
    save_dir = "save"
    eval_flag = True
    eval_period = 10
    num_workers = 4
    batchsize = 1
    total_epoch = 300
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank      = 0
    #---------------------------------------------------#
    #   获得类
    #---------------------------------------------------#
    def get_classes(classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

    #---------------------------------------------------#
    #   获得先验框
    #---------------------------------------------------#
    def get_anchors(anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors, len(anchors)

    class_names,num_classes = get_classes(classes_path)
    model = YoloBody(anchors_mask,num_classes,pretrained = pretrained)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。3[0m")
    for param in model.backbone.parameters():
            param.requires_grad = False
    model_train = model.train()
    model_train = torch.nn.DataParallel(model)
    model_train = model_train.cuda()

    for param in model.backbone.parameters():
            param.requires_grad = False
    yolo_loss = YOLOLoss(anchors=anchors,num_classes=num_classes,input_shape=input_shape)
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    nbs             = 64
    lr_decay_type       = "cos"
    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit     = min(max(batchsize / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batchsize / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    pg0, pg1, pg2 = [], [], []  
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)    
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)   
    optimizer = {
        'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
        'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
    }[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, 300)
    
    epoch_step = num_train//batchsize
    epoch_step_val = num_val//batchsize

    train_dataset = YOLOdataloader(train_lines,input_shape=input_shape,num_classes = num_classes,train = True)
    val_dataset = YOLOdataloader(val_lines,input_shape=input_shape,num_classes = num_classes,train = True)
    #TODO:这里没有使用fn函数，看看能不能跑
    gen = torch.utils.data.DataLoader(train_dataset,batch_size = batchsize,shuffle = True,
                                        num_workers = num_workers,pin_memory = True,drop_last = True,collate_fn = yolo_dataset_collate)
    gen_val = torch.utils.data.DataLoader(val_dataset,batch_size = batchsize,shuffle = True,
                                            num_workers = num_workers,pin_memory = True,drop_last = True,collate_fn = yolo_dataset_collate)
        
    # 开始训练
    with open("logs/log.txt","w") as f:
        for epoch in range(total_epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            for iter,batch in enumerate(gen):
                imgs,boxes = batch
                boxes = [ann.cuda() for ann in boxes]
                optimizer.zero_grad()
                with torch.no_grad():
                    imgs = imgs.cuda()
                model_train.train()
                predicts = model_train(imgs)
                loss = 0
                for l,predict in enumerate(predicts):
                    loss += yolo_loss(l=l,input = predict,targets = boxes,imgs=imgs)
                loss.backward()
                optimizer.step()
                print("epoch: {} iter:{}/{} loss:{}".format(epoch,iter,epoch_step,loss))
                f.writelines("epoch: {} iter:{}/{} loss:{}\n".format(epoch,iter,epoch_step,loss))
            
            if (epoch + 1) % save_period == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f.pth" % (epoch + 1, loss)))



    
    