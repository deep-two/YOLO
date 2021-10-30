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


class YOLO(nn.Module):
    def __init__(self, anchor_info, classes):
        super().__init__()
        self.anchor_info = anchor_info
        self.classes = classes
        self.conv_layer  = nn.Conv2d()

    def reorganize(self, feature):
        feature1 = feature[:,:,0:13,0:13]
        feature2 = feature[:,:,0:13,13:26]
        feature3 = feature[:,:,13:26,0:13]
        feature4 = feature[:,:,13:26,13:26]
        return torch.cat([feature1,feature2, feature3, feature4], axis=1)

    def forward(self, featuremap, output):
        reorg_map = self.reorganize(featuremap)
        new_feature_map = torch.cat([output, reorg_map], axis=1) # [batch, 3072, 13, 13] when input_size: 416
        output = self.conv_layer(new_feature_map)
        return output



if __name__ == "__main__":
    _in = torch.FloatTensor([[[[0 for _ in range(2**6 * 13)] for _ in range(2**6 * 13)] for _ in range(3)] for _ in range(2)])

    model = Darknet(_in.shape[2], pretrain=True)
    output = model(_in)

    print(output.shape)
