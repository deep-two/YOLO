import torch
import torch.nn as nn

class Darknet(nn.Module):
    def __init__(self) -> None:
        super(Darknet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(num_features=32))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(num_features=64))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(num_features=128))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(num_features=256))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(num_features=512))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding='same'),
                                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(num_features=1024))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x

if __name__ == "__main__":
    _in = torch.FloatTensor([[[[0 for _ in range(2**6 * 13)] for _ in range(2**6 * 13)] for _ in range(3)] for _ in range(2)])

    model = Darknet()
    output = model(_in)

    print(output.shape)
