import os
import sys
import numpy as np

# VOC_CLASSES = [
#             "background",
#             "aeroplane",
#             "bicycle",
#             "bird",
#             "boat",
#             "bottle",
#             "bus",
#             "car",
#             "cat",
#             "chair",
#             "cow",
#             "diningtable",
#             "dog",
#             "horse",
#             "motorbike",
#             "person",
#             "potted plant",
#             "sheep",
#             "sofa",
#             "train",
#             "tv/monitor",
#         ]

VOC_CLASSES = [
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

VOC_COLORS = np.random.randint(0, 255, size=(80, 3), dtype='uint8')


ANCHOR_BOX_SIZE = [
    (159.5435024065161, 256.1136616068123),
    (374.98336557059963, 333.32069632495165),
    (33.865758320303776, 43.12776412776413),
    (94.72419468610393, 117.84363978368211),
    (314.60921501706486, 157.32821387940842)
]
