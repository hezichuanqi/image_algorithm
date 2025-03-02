import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
        pass



class IOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(IOULoss, self).__init__()
        self.reduction = reduction
        pass
