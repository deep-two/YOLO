import torch
import torch.nn as nn


class Darknet(nn.Module):
    def __init__(self, img_shape, pretrain=False) -> None:
        super(Darknet, self).__init__()
        self.pretrain = pretrain
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=32), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=64), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=128), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=256), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=512))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=1024))
        

        if self.pretrain is True:
            assert img_shape/(2**6) == int(img_shape/(2**6))
            self.pretrain_layer = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding='same'),
                                                nn.AvgPool2d(kernel_size=int(img_shape/(2**6))))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5_= self.maxpool(x5)
        x6 = self.conv6(x5_)
        

        if self.pretrain is True:
            x = self.pretrain_layer(x)
        
        # x5 for fine_grained_feature, x6 is output
        return x5, x6



class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        b, c, h, w = x.size()
        out_w, out_h, out_c = int(w / stride), int(h / stride), c * (stride * stride)

        x = reorganize(x, out_h, out_w)
        return x



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
        print(conv1.shape)
        conv2 = self.conv2(conv1)
        print(conv2.shape)
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



def reorganize(feature, out_h, out_w):
    feature1 = feature[:,:, 0:out_h, 0:out_w]
    feature2 = feature[:,:, 0:out_h, out_w:]
    feature3 = feature[:,:, out_h:, 0:out_w]
    feature4 = feature[:,:, out_h:, out_w:]
    return torch.cat([feature1,feature2, feature3, feature4], axis=1)



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


if __name__ == "__main__":
    _in = torch.FloatTensor([[[[0 for _ in range(2**6 * 13)] for _ in range(2**6 * 13)] for _ in range(3)] for _ in range(2)])

    model = Darknet(_in.shape[2], pretrain=True)
    output = model(_in)

    print(output.shape)
