import collections
import typing
import xml.etree.ElementTree as ET
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from utils.config import cfg

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

COLORS = np.random.randint(0, 255, size=(80, 3), dtype='uint8')


class PascalVOCDataset(VOCDetection):
    def __init__(self, root="./data/pascal_voc",
                 image_set="train",
                 year="2012",
                 download=False,
                 transform=None,
                 input_size=416) :
        self.input_size = (input_size, input_size)
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # path, label = self.samples[index]
        pillow_image = Image.open(self.images[index]).convert('RGB')
        image = np.array(pillow_image.resize(self.input_size))
        # image = cv2.imread(self.images[index])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        targets = []
        labels = []

        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox']['ymax'], cfg.DATA.CLASSES.index(t['name'])

            # targets.append(list(label[:4])) # 바운딩 박스 좌표
            # labels.append(label[4])         # 바운딩 박스 클래스

            targets.append(label)

        if self.transforms:
            augmentations = self.transforms(image=image, bboxes=targets)
            image = augmentations['image']
            targets = augmentations['bboxes']

        # targets = np.concatenate(targets, axis=0)
        targets = np.array(targets)

        return_dict = {
            'img': image,
            'annot': targets
        }

        # return image, targets, labels
        return return_dict


    def collater(self, data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = torch.FloatTensor(annot)
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1
        imgs = torch.from_numpy(np.stack(imgs, axis=0)).to(torch.float32)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs, annot_padded


    def parse_voc_xml(self, node: ET.Element) -> Dict[str, typing.Any]:  # xml 파일을 dictionary로 반환
        voc_dict: Dict[str, typing.Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, typing.Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict



class ImageNetDataset(Dataset):
    """
    ImageNet class doesn't support ImageNet dataset from the kaggle(Link below).
    (https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)
    So we can't use ImageNet class that we made the custom Dataset class.
    """
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label