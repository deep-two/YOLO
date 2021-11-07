import torch.nn as nn
import numpy as np
import torch

from darknet import Darknet, ReorgLayer
from anchor_box import DirectLocation


class yolov2(nn.Module):
    def __init__(self, n_class, n_anchor, is_training = True):
        super(yolov2, self).__init__()
        self.darknet = Darknet()
        self.fineGrained = FineGrainedFeature(n_anchor, n_class)
        self.directlocation = DirectLocation()

        self.is_training = is_training

    def nms(dets, thresh):
        """
        Pure Python NMS baseline.
        reference: https://github.com/longcw/yolo2-pytorch/blob/master/utils/nms/py_cpu_nms.py
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def forward(self, input):
        x5, x6 = self.darknet(input)
        x = self.fineGrained(x5, x6)
        x = self.directlocation(x)

        if self.is_training:
            return x
        else:
            self.nms(x, 0.5)
            return


######지환#######

class YOLO(nn.Module):
    def __init__(self ):
        super().__init__()
        self.darknet = Darknet(416)
        self.reorglayer = ReorgLayer(stride=2)
        
        self.conv1, c1 = _make_layers((512*(2*2) + 1024), [(1024, 3)])
        self.conv2 = Conv2d(c1, 125, 1, 1, relu=False)
        self.global_average_pool = nn.AvgPool2d((1, 1))
        
        # train
        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss = None


    def loss(self):
        return self.bbox_loss + self.iou_loss + self.cls_loss


    def forward(self, x):
        fine, out = self.darknet(x)
        fine = self.reorglayer(fine)
        feature_map = torch.cat([fine, out], axis=1)
        conv1 = self.conv1(feature_map)
        conv2 = self.conv2(conv1)
        
        global_average_pool = self.global_average_pool(conv2)

        bsize, _, h, w = global_average_pool.size()
        
        # batch, _, anchors, class_score(20) + iou_score(1) + bbox_pred(4)
        global_average_pool_reshaped = global_average_pool.permute(0, 2, 3, 1).contiguous().view(bsize, -1, 5, 20 + 5)  

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = torch.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)                          # (x,y,w,h)
        iou_pred = torch.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])

        score_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()  
        prob_pred = torch.softmax(score_pred.view(-1, score_pred.size()[-1]), dim=1).view_as(score_pred) 
        


        #### loss 부분 추가해야함 #####


        return bbox_pred, iou_pred, prob_pred




########### conv_layers ##############

def _make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(Conv2d_BatchNorm(in_channels,
                                                out_channels,
                                                ksize,
                                                same_padding=True))
                # layers.append(net_utils.Conv2d(in_channels, out_channels,
                #     ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



#config 필요
# label_names = ('aeroplane', 'bicycle', 'bird', 'boat',
#                'bottle', 'bus', 'car', 'cat', 'chair',
#                'cow', 'diningtable', 'dog', 'horse',
#                'motorbike', 'person', 'pottedplant',
#                'sheep', 'sofa', 'train', 'tvmonitor')
# num_classes = len(label_names)

# anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
#                       (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)],
#                      dtype=np.float)
# num_anchors = len(anchors)

