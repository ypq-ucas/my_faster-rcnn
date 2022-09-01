from utils.utils import show_box
import torch
import numpy as np
array = np.ones([416,416,3])*150
boxes = [[185,62,279,199],[90,78,403,336]]
show_box(array,boxes)

#我写的
model = torch.nn.Dataparallel(model)
model.train()
output = model(img)

#作者写的
model_train = model.train()
model_train = torch.nn.Dataparallel(model)
model_train.train()
output = model_train(img)
