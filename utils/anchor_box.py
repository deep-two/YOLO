import numpy as np
import matplotlib.pyplot as plt
from torch import rand, sigmoid, exp, nn

from model.loss import bbox_iou
from utils.datasets import PascalVOCDataset

np.random.seed(42)


class DimensionCluster:
    def __init__(self, n_clusters: int):
        self.n_clusters: int = n_clusters
        self.boxes = self.get_bbox()
        self.group, self.anchor_box = self.kmeans()
        # self.visualize_cluster()

    def get_bbox(self):
        # Load Bbox Dataset
        boxes = PascalVOCDataset().get_bbox()
        boxes = np.array(boxes)
        boxes[:, 2] = (boxes[:, 2]-boxes[:, 0])
        boxes[:, 3] = (boxes[:, 3]-boxes[:, 1])
        boxes[:, 0] = 0
        boxes[:, 1] = 0
        return boxes

    def kmeans(self) -> np.array:
        iteration = 0
        max_iteration = 15000
        loss_convergence = 1e-6
        n_boxes: int = len(self.boxes)
        # choice init_clusters
        centroids = self.boxes[np.random.choice(n_boxes, self.n_clusters, replace=False)]
        old_loss = 0
        while True:
            loss = 0
            groups = []
            new_centroid = []

            for i in range(self.n_clusters):
                groups.append([])
                new_centroid.append([0, 0, 0, 0])

            for box in self.boxes:
                min_distance = 1
                group_index = 0
                for centroid_index, centroid in enumerate(centroids):
                    distance = 1-bbox_iou(box, centroid)
                    if distance < min_distance:
                        min_distance = distance
                        group_index = centroid_index
                groups[group_index].append(box)
                loss += min_distance
                new_centroid[group_index][2] += box[2]
                new_centroid[group_index][3] += box[3]

            for i in range(self.n_clusters):
                new_centroid[i][2] /= len(groups[i])
                new_centroid[i][3] /= len(groups[i])

            centroids = new_centroid
            iteration += 1
            loss /= self.boxes.shape[0]
            print(loss)
            if abs(old_loss-loss) < loss_convergence or iteration > max_iteration:
                break
            old_loss = loss

        for centroid in centroids:
            print(centroid[2], centroid[3])
        return groups, centroids

    def visualize_cluster(self):
        self.anchor_box = np.array(self.anchor_box)
        colors = ['red', 'blue', 'yellow', 'green', 'grey']
        for i in range(len(self.group)):
            for j in range(len(self.group[i])):
                plt.scatter(x=self.group[i][j][2], y=self.group[i][j][3], color=colors[i])
        # plt.scatter(x=self.boxes[:, 2], y=self.boxes[:, 3])
        plt.scatter(x=self.anchor_box[:, 2], y=self.anchor_box[:, 3], c='red')
        plt.show()

    @property
    def get_anchor_box_size(self):
        return self.anchor_box


class DirectLocation(nn.Module):
    def __init__(self, centroid):
        super().__init__()
        self.anchor_box = self.get_anchor_box()
        self.n_anchor = len(self.anchor_box)

    def location_prediction(self, c_x=0, c_y=0):
        anchor_box = []
        for anchor in self.anchor_box:
            p_w = anchor[2]
            p_h = anchor[3]
            locate_prediction = rand(5, requires_grad=True)

            b_x = sigmoid(locate_prediction[0]) + c_x
            b_y = sigmoid(locate_prediction[1]) + c_y
            b_w = p_w * exp(locate_prediction[2])
            b_h = p_h * exp(locate_prediction[3])
            locate_prediction[4] = bbox_iou()
            anchor_box.append([b_x, b_y, b_w, b_h, locate_prediction[4]])

    def get_anchor_box(self):
        dimension_cluster = DimensionCluster(n_clusters=5)
        anchor_box = dimension_cluster.kmeans()
        return anchor_box


def main():
    DimensionCluster(n_clusters=5)

