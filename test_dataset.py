from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import DataLoader
from utils.datasets import PascalVOCDataset
import matplotlib.pyplot as plt
import numpy as np

CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COLORS = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

# 시각화 함수
def show(img, targets, labels):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)
    W, H = img.size

    for tg,label in zip(targets, labels):
        id_ = int(label) # class
        bbox = tg[:4]    # [x1, y1, x2, y2]

        color = [int(c) for c in COLORS[id_]]
        name = CLASSES[id_]

        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=tuple(color), width=3)
        draw.text((bbox[0], bbox[1]), name, fill=(255,255,255,0))
    plt.imshow(np.array(img))


pascal_train_dataset = PascalVOCDataset(image_set="train", root="./data/pascal_voc")
plt.figure(figsize=(10,10))

print(pascal_train_dataset[0])

plt.show()

pascal_train_dataloader = DataLoader(
    pascal_train_dataset, batch_size=64, shuffle=True
)

print()
