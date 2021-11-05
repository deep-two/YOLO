from darknet import Darknet, FineGrainedFeature
from anchor_box import DirectLocation
import torch.nn as nn
import numpy as np

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
