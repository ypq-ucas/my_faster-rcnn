import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw,Image
from utils.utils import show_box,get_classes
def show_box(l,x,y,w,h,conf,imgs,stride_h,stride_w,scaled_anchors,pred_cls):
    class_names,class_num = get_classes("voc_classes.txt")
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]][l]
    imgs = imgs[0].permute(1,2,0).cpu()
    imgs = np.uint8(imgs)
    imgs = Image.fromarray(imgs)
    draw = ImageDraw.Draw(imgs)
    x,y,w,h,conf = x[0],y[0],w[0],h[0],conf[0]
    if(torch.sum(conf>0.5)):
        boxes = torch.nonzero(conf>0.5)
        for box in boxes:
            x = x[box[0]][box[1]][box[2]]*stride_h+box[2]*stride_h
            y = y[box[0]][box[1]][box[2]]*stride_h+box[1]*stride_w
            w = torch.exp(w[box[0]][box[1]][box[2]])*scaled_anchors[anchors_mask[box[0]]][0]*stride_w
            h = torch.exp(h[box[0]][box[1]][box[2]])*scaled_anchors[anchors_mask[box[0]]][1]*stride_w
            x0,y0 = x-w/2,y-h/2
            x1,y1 = x+w/2,y+h/2
            pred_cls = pred_cls[0][box[0]][box[1]][box[2]]
            pred_cls = torch.argmax(pred_cls)
            class_name = class_names[(pred_cls)]
            label = '{} {:.2f}'.format(class_name, conf[box[0]][box[1]][box[2]])
            label_size = draw.textsize(label)
            label = label.encode('utf-8')
            draw.rectangle([x0,y0, x0+label_size[0],y0+label_size[1]])
            draw.text([x0,y0], str(label,'UTF-8'), fill=(0, 0, 0),)
            draw.rectangle([x0,y0,x1,y1],outline="red")
            del draw
        imgs.show()



def show_y_true(y_true,imgs,stride_h,stride_w,scaled_anchors,l,targets):
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
    imgs = imgs[0].permute(1,2,0).cpu()
    imgs = np.uint8(imgs)
    imgs = Image.fromarray(imgs)
    draw = ImageDraw.Draw(imgs)
    targets = targets[0]
    for target in targets:
        x0,y0 = target[0]-target[2]/2,target[1]-target[3]/2
        x1,y1 = target[0]+target[2]/2,target[1]+target[3]/2
        x0,y0,x1,y1 = x0*416+5,y0*416+5,x1*416,y1*416
        draw.rectangle([x0,y0,x1,y1],outline="red")

    y_true = y_true[0]
    obj_mask = torch.nonzero(y_true[...,4])
    if(torch.sum(y_true[...,4])):
        for index in obj_mask:
            anchor = scaled_anchors[anchors_mask[l][index[0]]]
            box = y_true[index[0]][index[1]][index[2]]
            x,y = box[0]*stride_h,box[1]*stride_h
            x+=index[2]*stride_h
            y+=index[1]*stride_h
            w,h = box[2]*stride_h*anchor[0],box[3]*stride_h*anchor[1]
            x0,x1 = x-w/2,x+w/2
            y0,y1 = y-h/2,y+h/2
            box = (x0,y0,x1,y1)
            draw.rectangle(box)
        imgs.show()
        del draw




class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda =True, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13?????????????????????anchor???[116,90],[156,198],[373,326]
        #   26x26?????????????????????anchor???[30,61],[62,45],[59,119]
        #   52x52?????????????????????anchor???[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.giou           = False
        self.balance        = [0.4, 1.0, 4]
        self.box_ratio      = 0.05
        self.obj_ratio      = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio      = 1 * (num_classes / 80)

        self.ignore_threshold = 0.5
        self.cuda           = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        """
        ????????????
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        ????????????
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        #----------------------------------------------------#
        #   ?????????????????????????????????
        #----------------------------------------------------#
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        #----------------------------------------------------#
        #   ?????????????????????????????????
        #----------------------------------------------------#
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half

        #----------------------------------------------------#
        #   ?????????????????????????????????iou
        #----------------------------------------------------#
        intersect_mins  = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area

        #----------------------------------------------------#
        #   ?????????????????????????????????????????????????????????
        #----------------------------------------------------#
        enclose_mins    = torch.min(b1_mins, b2_mins)
        enclose_maxes   = torch.max(b1_maxes, b2_maxes)
        enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        #----------------------------------------------------#
        #   ?????????????????????
        #----------------------------------------------------#
        enclose_area    = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou            = iou - (enclose_area - union_area) / enclose_area
        
        return giou
        
    def forward(self, l, input,imgs = None, targets=None):
        #----------------------------------------------------#
        #   l?????????????????????????????????????????????????????????????????????????????????
        #   input???shape???  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets?????????????????????,?????????xywh???class???xywh???????????????????????????????????????
        #----------------------------------------------------#
        #--------------------------------#
        #   ??????????????????????????????????????????
        #   13???13
        #--------------------------------#
        bs      = input.size(0)#batchsize
        in_h    = input.size(2)#height
        in_w    = input.size(3)#width
        #-----------------------------------------------------------------------#
        #   ????????????
        #   ????????????????????????????????????????????????????????????
        #   ??????????????????13x13??????????????????????????????????????????????????????32*32????????????
        #   ??????????????????26x26??????????????????????????????????????????????????????16*16????????????
        #   ??????????????????52x52??????????????????????????????????????????????????????8*8????????????
        #   stride_h = stride_w = 32???16???8
        #   stride_h???stride_w??????32???
        #-----------------------------------------------------------------------#
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        #-------------------------------------------------#
        #   ???????????????scaled_anchors??????????????????????????????
        #-------------------------------------------------#
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]#anchor????????????????????????
        #-----------------------------------------------#
        #   ?????????input???????????????????????????shape?????????
        #   bs, 3*(5+num_classes), 13, 13 => batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #-----------------------------------------------#
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        #-----------------------------------------------#
        #   ???????????????????????????????????????
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   ??????????????????????????????
        #-----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]
        #-----------------------------------------------#
        #   ?????????????????????????????????
        #-----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        #   ???????????????
        #-----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #-----------------------------------------------#
        #   ????????????????????????????????????
        #-----------------------------------------------#
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)
        # show_y_true(y_true,imgs,stride_h,stride_w,scaled_anchors,l,targets = targets)
        #---------------------------------------------------------------#
        #   ???????????????????????????????????????????????????????????????????????????
        #   ?????????????????????????????????????????????????????????????????????????????????????????????
        #   ????????????????????????
        #----------------------------------------------------------------#
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)
        #y_true:tx,tx,w,h,conf,cls

        if self.cuda:
            y_true          = y_true.type_as(x)
            noobj_mask      = noobj_mask.type_as(x)
            box_loss_scale  = box_loss_scale.type_as(x)
        #--------------------------------------------------------------------------#
        #   box_loss_scale??????????????????????????????????????????0-1???????????????????????????0-1?????????
        #   2-??????????????????????????????????????????????????????????????????????????????
        #--------------------------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale
            
        loss        = 0
        obj_mask    = y_true[..., 4] == 1
        n           = torch.sum(obj_mask)
        if n != 0:
            if self.giou:
                #---------------------------------------------------------------#
                #   ????????????????????????????????????giou
                #----------------------------------------------------------------#
                giou        = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
                loss_loc    = torch.mean((1 - giou)[obj_mask])
            else:
                #-----------------------------------------------------------#
                #   ???????????????????????????loss?????????BCELoss???????????????
                #-----------------------------------------------------------#
                loss_x      = torch.mean(self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale[obj_mask])
                loss_y      = torch.mean(self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale[obj_mask])
                #-----------------------------------------------------------#
                #   ????????????????????????loss
                #-----------------------------------------------------------#
                loss_w      = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale[obj_mask])
                loss_h      = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale[obj_mask])
                loss_loc    = (loss_x + loss_y + loss_h + loss_w) * 0.1

            loss_cls    = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss        += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        # loss_conf   = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss_conf   = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf)))
        loss        += loss_conf * self.balance[l] * self.obj_ratio
        show_box(l,x,y,w,h,conf,imgs,stride_h,stride_w,scaled_anchors,pred_cls)
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   ???????????????????????????????????????
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   ?????????????????????????????????????????????????????????
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   ???????????????????????????????????????????????????????????????
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A????????????????????????B?????????????????????
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   ??????????????????
        #-----------------------------------------------------------#
        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   ??????????????????????????????????????????
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   ???IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]
    
    def get_target(self, l, targets, anchors, in_h, in_w):
        
        #-----------------------------------------------------#
        #   ??????????????????????????????
        #-----------------------------------------------------#
        bs              = len(targets)
        noobj_mask      = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        
        #-----------------------------------------------------#
        #-----------------------------------------------------#
        #   ?????????????????????????????????
        #-----------------------------------------------------#
        box_loss_scale  = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        #-----------------------------------------------------#
        y_true          = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        for b in range(bs):            
            if len(targets[b])==0:
                continue
            batch_target = torch.zeros_like(targets[b])
            #-------------------------------------------------------#
            #   ?????????????????????????????????????????????
            #-------------------------------------------------------#
            batch_target[:, [0,2]] = targets[b][:, [0,2]]*in_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]]*in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            
            #-------------------------------------------------------#
            #   ??????????????????????????????
            #   num_true_box, 4
            #-------------------------------------------------------#
            gt_box          = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            #-------------------------------------------------------#
            #   ??????????????????????????????
            #   9, 4
            #-------------------------------------------------------#
            anchor_shapes   = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            #-------------------------------------------------------#
            #   ???????????????
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]?????????????????????9???????????????????????????
            #   best_ns:
            #   [?????????????????????????????????max_iou, ????????????????????????????????????????????????]
            #-------------------------------------------------------#
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                #----------------------------------------#
                #   ????????????????????????????????????????????????????????????
                #----------------------------------------#
                k = self.anchors_mask[l].index(best_n)
                #----------------------------------------#
                #   ????????????????????????????????????
                #----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                #----------------------------------------#
                #   ????????????????????????
                #----------------------------------------#
                c = batch_target[t, 4].long()
                noobj_mask[b, k, j, i] = 0

                #----------------------------------------#
                #   tx???ty????????????????????????????????????
                #----------------------------------------#
                if not self.giou:
                    #----------------------------------------#
                    #   tx???ty????????????????????????????????????
                    #----------------------------------------#
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                else:
                    #----------------------------------------#
                    #   tx???ty????????????????????????????????????
                    #----------------------------------------#
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                #----------------------------------------#
                #   ????????????xywh?????????
                #   ?????????loss?????????????????????loss?????????
                #----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        #-----------------------------------------------------#
        #   ??????????????????????????????
        #-----------------------------------------------------#
        bs = len(targets)

        #-----------------------------------------------------#
        #   ????????????????????????????????????????????????
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # ????????????????????????
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        #-------------------------------------------------------#
        #   ??????????????????????????????????????????
        #-------------------------------------------------------#
        pred_boxes_x    = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y    = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        
        for b in range(bs):           
            #-------------------------------------------------------#
            #   ?????????????????????????????????
            #   pred_boxes_for_ignore      num_anchors, 4
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            #-------------------------------------------------------#
            #   ?????????????????????????????????????????????????????????????????????
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                #-------------------------------------------------------#
                #   ?????????????????????????????????????????????
                #-------------------------------------------------------#
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)
                #-------------------------------------------------------#
                #   ???????????????
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   ????????????????????????????????????????????????
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

