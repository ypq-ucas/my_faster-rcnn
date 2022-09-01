import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
class RegionProposalNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()