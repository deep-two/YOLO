import cv2
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class PascalVOCDataset(data.Datset):
    def __init__(self, root="~/data/pascal_voc", image_set="train", download=True, transform=None):

        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label



class ImageNetDataset(torchvision.datasets.ImageNet):
    def __init__(self, root="~/data/imagenet_2012", image_set="train", download=True, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
