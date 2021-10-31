import cv2
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageNet, VOCDetection
from torch.utils.data import DataLoader

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class PascalVOCDataset(VOCDetection):
    def __init__(self, root="/Users/jin-yejin/Documents/deep2/YOLO/data/pascal_voc", image_set="train", download=False, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    # def __len__(self):
    #     return len(self.files[self.split])

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label



class ImageNetDataset(ImageNet):
    def __init__(self, root="~/data/imagenet", image_set="train", download=False, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

# test = ImageNetDataset()
pascal_train_dataset = PascalVOCDataset()
imagenet_train_dataset = ImageNetDataset()

print(pascal_train_dataset)
print(imagenet_train_dataset)

pascal_train_loader = DataLoader(
    pascal_train_dataset, batch_size=64, shuffle=True
)

imagenet_train_loader = DataLoader(
    imagenet_train_dataset, batch_size=64, shuffle=True
)

print(pascal_train_dataset)
print(imagenet_train_loader)