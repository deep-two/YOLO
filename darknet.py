import torch
import torch.nn as nn

class Darknet(nn.Module):
    def __init__(self, img_shape, pretrain=False) -> None:
        super(Darknet, self).__init__()
        self.pretrain = pretrain

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
                                    nn.BatchNorm2d(num_features=512), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.BatchNorm2d(num_features=1024))
        self.conv6_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.pretrain is True:
            assert img_shape/(2**6) == int(img_shape/(2**6))
            self.pretrain_layer = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding='same'),
                                                nn.AvgPool2d(kernel_size=int(img_shape/(2**6))))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x6 = self.conv6(x)
        x = self.conv6_maxpool(x6)

        if self.pretrain is True:
            x = self.pretrain_layer(x)

        return x6, x

if __name__ == "__main__":
    _in = torch.FloatTensor([[[[0 for _ in range(2**6 * 13)] for _ in range(2**6 * 13)] for _ in range(3)] for _ in range(2)])

    model = Darknet(_in.shape[2], pretrain=True)
    output = model(_in)

    print(output.shape)
