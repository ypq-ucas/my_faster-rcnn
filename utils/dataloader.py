import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
class YOLOdataloader(Dataset):
    def __init__(self,data_lines,input_shape,num_classes,train) -> None:
        super().__init__()
        self.data_lines = data_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lenth = len(data_lines)
        self.train = train
    def __len__(self):
        return self.lenth
    def __getitem__(self, index):
        index = index%self.lenth
        data_line = self.data_lines[index].split()
        image = Image.open(data_line[0]).convert("RGB")
        origin_image = image
        box = np.array([np.array(list(map(int,box.split(",")))) for box in data_line[1:]])
        #将图像大小重塑，上 下加上灰条
        image = self.resize_image(image)
        image = torch.tensor(image)
        #将目标框同样重塑,且以xywh的形式输出
        box = self.replace_box(origin_image,box)
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
        image = image.permute(2,0,1).tolist()
        return image,box
    def resize_image(self,image):
        #将图像大小重塑，上下加上灰条
        iw,ih = image.size
        h,w = self.input_shape
        scale = min(w/iw,h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        #给图像的多余部分加上灰条
        image = image.resize((nw,nh),Image.BICUBIC)
        new_image = Image.new("RGB",(w,h),(128,128,128))
        new_image.paste(image,(dx,dy))
        image = np.array(new_image)
        return image
    def replace_box(self,image,box):
        iw,ih = image.size
        h,w = self.input_shape
        scale = min(w/iw,h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        np.random.shuffle(box)
        box[:,[0,2]] = box[:,[0,2]]*nw/iw+dx
        box[:,[1,3]] = box[:,[1,3]]*nw/iw+dy
        box[:,0:2][box[:,0:2]<0] = 0
        box[:,2][box[:,2]>w] = w
        box[:,3][box[:,3]>h] = h
        box_w = box[:,2]-box[:,0]
        box_h = box[:,3]-box[:,1]
        box = box[np.logical_and(box_w>1,box_h>1)]
        x = torch.tensor((box[:,0]+box[:,2])//2)
        y = torch.tensor((box[:,1]+box[:,3])//2)
        w = torch.tensor(box[:,2]-box[:,0])
        h = torch.tensor(box[:,3]-box[:,1])
        box[:,0] =x
        box[:,1] =y
        box[:,2] =w
        box[:,3] =h
        return box
def yolo_dataset_collate(batch):
        images = []
        bboxes = []
        for img, box in batch:
            images.append(img)
            bboxes.append(box)
        images = torch.tensor(images).type(torch.FloatTensor)
        bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        return images, bboxes










