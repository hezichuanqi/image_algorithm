import torch

def iou(boxes1, boxes2,eps=1e-6):
    """
    boxes1: gt框，[N, 4]-->(x1, y1, x2, y2)
    boxes2: 预测框，[M, 4]-->(xd1,yd1,xd2,yd2)
    
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    #取出左上右下坐标
    xg1, yg1, xg2, yg2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3] # (N, )
    xp1, yp1, xp2, yp2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3] # (M, )
    #维度扩展
    xg1, yg1 = xg1.view(N, 1).expand(N, M), yg1.view(N, 1).expand(N, M) # (N, M)
    xg2, yg2 = xg2.view(N, 1).expand(N, M), yg2.view(N, 1).expand(N, M)
    xp1, yp1 = xp1.view(1, M).expand(N, M), yp1.view(1, M).expand(N, M)# (N, M)
    xp2, yp2 = xp2.view(1, M).expand(N, M), yp2.view(1, M).expand(N, M)
    #计算预测框和真实框的交集
    xmin = torch.max(xg1, xp1) # (N, M)，左上点的x坐标
    ymin = torch.max(yg1, yp1)
    xmax = torch.min(xg2, xp2)
    ymax = torch.min(yg2, yp2)

    #计算面积
    area_g = (xg2 - xg1) * (yg2 - yg1)
    area_p = (xp2 - xp1) * (yp2 - yp1)
    area_c = (xmax - xmin) * (ymax - ymin)

    #计算iou
    iou = area_c / (area_g + area_p - area_c + eps) # (N, M)

    return iou

def Giou(boxes1, boxes2,eps=1e-6):
    """
    boxes1: gt框，[N, 4]-->(x1, y1, x2, y2)
    boxes2: 预测框，[M, 4]-->(xd1,yd1,xd2,yd2)"
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    #取出左上右下坐标
    xg1, yg1, xg2, yg2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3] # (N, )
    xp1, yp1, xp2, yp2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3] # (M, )
    #维度扩展
    xg1, yg1 = xg1.view(N, 1).expand(N, M), yg1.view(N, 1).expand(N, M) # (N, M)
    xg2, yg2 = xg2.view(N, 1).expand(N, M), yg2.view(N, 1).expand(N, M)
    xp1, yp1 = xp1.view(1, M).expand(N, M), yp1.view(1, M).expand(N, M)# (N, M)
    xp2, yp2 = xp2.view(1, M).expand(N, M), yp2.view(1, M).expand(N, M)


    #计算面积
    area_g = (xg2 - xg1) * (yg2 - yg1)
    area_p = (xp2 - xp1) * (yp2 - yp1)

    #计算iou
    iou_g = iou(boxes1, boxes2) # (N, M)

    #计算最小外接矩形的左上右下坐标
    xmin = torch.min(xg1, xp1)
    ymin = torch.min(yg1, yp1)
    xmax = torch.max(xg2, xp2)
    ymax = torch.max(yg2, yp2)

    #计算最小外接矩形的面积
    area_u = (xmax - xmin) * (ymax - ymin)

    #计算giou
    giou = iou_g - (area_u - area_g - area_p + eps) / (area_u + eps)

    return giou


if __name__ == "__main__":
    
    boxes1 = torch.randn(10, 4)
    boxes2 = torch.randn(10, 4)
    print(iou(boxes1, boxes2))
