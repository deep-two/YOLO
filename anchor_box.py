import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from torch import rand, sigmoid, exp
from torchvision import transforms

from loss import bbox_iou


class DimensionCluster:
    def __init__(self, n_clusters: int):
        self.n_clusters: int = n_clusters
        self.boxes: List = self.get_bbox()

    def get_bbox(self) -> List:
        # Load Bbox Dataset

        # Iteration Dataset and Get bbox width and height
        boxes = []
        width
        height
        boxes.append([width, height])
        return boxes

    def kmeans(self) -> np.array:
        n_boxes: int = len(self.boxes)
        # distances = np.empty((n_boxes, self.n_clusters))
        last_nearest = np.zeros((n_boxes, ))

        # choice init_clusters
        clusters = boxes[np.random.choice(box_number, self.n_clusters, replace=False)]

        while True:
            distances = 1-self.bbox_iou(boxes, clutsers)
            centroids = np.argmin(distances, axis=1)

            if (last_nearest == centroids).all():
                break

            for cluster in range(self.n_clusters):
                clusters[cluster] = np.median(boxes[centroids == cluster], axis=0)

            last_nearest = centroids

        return clusters


class DirectLocation:
    def __init__(self, centroid):
        self.anchor_box = self.get_anchor_box()
        pass

    def location_prediction(self, c_x=0, c_y=0):
        locate_prediction = rand(5, requires_grad=True)
        b_x = sigmoid(locate_prediction[0]) + c_x
        b_y = sigmoid(locate_prediction[1]) + c_y
        b_w = p_w * exp(locate_prediction[2])
        b_h = p_h * exp(locate_prediction[3])
        bbox = [b_x, b_y, b_w, b_h]
        locate_prediction[4] = bbox_iou()

    def get_anchor_box(self):
        anchor_box = DimensionCluster(n_clusters=5)
        return anchor_box


main()
