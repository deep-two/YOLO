from darknet import Darknet, FineGrainedFeature
from anchor_box import DirectLocation
import torch.nn as nn

class yolov2(nn.Module):
    def __init__(self, n_class, n_anchor, is_training = True):
        super(yolov2, self).__init__()
        self.darknet = Darknet()
        self.fineGrained = FineGrainedFeature(n_anchor, n_class)
        self.directlocation = DirectLocation()

        self.is_training = is_training

    def forward(self, input):
        x5, x6 = self.darknet(input)
        x = self.fineGrained(x5, x6)
        x = self.directlocation(x)

        if self.is_training:
            return x
        else:
            # needs nms
            return
