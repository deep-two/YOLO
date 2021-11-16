from cv2 import moments
import torch

def bbox_iou(inbox1, inbbox2):
    bbox1 = [0,0,0,0]
    x, y, w, h = inbox1
    bbox1[0] = x  - w / 2
    bbox1[1] = y  - h / 2
    bbox1[2] = x  + w / 2
    bbox1[3] = y  + h / 2

    bbox2 = [0,0,0,0]
    x, y, w, h = inbbox2
    bbox2[0] = x  - w / 2
    bbox2[1] = y  - h / 2
    bbox2[2] = x  + w / 2
    bbox2[3] = y  + h / 2

    left_x = max(bbox1[0], bbox2[0])
    top_y = max(bbox1[1], bbox2[1])
    right_x = min(bbox1[2], bbox2[2])
    bottom_y = min(bbox1[3], bbox2[3])

    if right_x - left_x < 0 or bottom_y - top_y < 0:
        return 0

    intersection = (right_x - left_x) * (bottom_y - top_y)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union

def match(outputs, gts):
    result = -torch.ones(outputs.shape)
    res_iou = -torch.ones(outputs.shape[0])
    thresh = 0.5

    for gt in gts:
        max_iou_idx = -1
        max_iou = -1
        for i, output in enumerate(outputs):
            iou = bbox_iou(gt[:4], output[:4])
            if max_iou < iou and iou > thresh:
                max_iou = iou
                max_iou_idx = i
        if max_iou_idx > -1:
            result[max_iou_idx] = gt
            res_iou[max_iou_idx] = max_iou
            
    return result, res_iou

def get_loss(outputs, gts, class_num=20, lambda_coordi=5, lambda_nobody = 0.5):
    loss_list = []
    for batch in range(outputs.shape[0]):
        pred_boxes = outputs[batch].view([-1,5+class_num])

        gt = gts[batch]
        gt_boxes = torch.zeros([len(gt), 5+class_num])
        gt_boxes[:, 4] = 1
        gt_boxes[:, 0] = (gt[:, 0] + gt[:, 2]) / 2 / 32
        gt_boxes[:, 1] = (gt[:, 1] + gt[:, 3]) / 2 /32
        gt_boxes[:, 2] = (gt[:, 2] - gt[:, 0]) / 32
        gt_boxes[:, 3] = (gt[:, 3] - gt[:, 1]) / 32
        gt_boxes[:, 5:] = torch.nn.functional.one_hot(gt[:, 4].to(torch.int64), num_classes=class_num)
        
        matched_gt, mask = match(pred_boxes, gt_boxes)
        print(mask)

        coordi_error = 0
        conf_error = 0
        class_error = 0
        for idx in range(len(mask)):
            if mask[idx] > 0:
                coordi_error += torch.square(pred_boxes[idx][0] - matched_gt[idx][0]) # x
                coordi_error += torch.square(pred_boxes[idx][1] - matched_gt[idx][1]) # y
                coordi_error += torch.square(torch.sqrt(pred_boxes[idx][2]) - torch.sqrt(matched_gt[idx][2])) # w
                coordi_error += torch.square(torch.sqrt(pred_boxes[idx][3]) - torch.sqrt(matched_gt[idx][3])) # h

                conf_error += torch.square(pred_boxes[idx][4] - matched_gt[idx][4])

                class_error += torch.sum(torch.square(pred_boxes[idx][5:] - matched_gt[idx][5:]))
            else:
                conf_error += torch.square(pred_boxes[idx][4]) * lambda_nobody

        error = lambda_coordi * coordi_error + conf_error + class_error
        loss_list.append(error)

    return sum(loss_list)

    
    


if __name__ == "__main__":
    # print(bbox_iou(torch.tensor([8,8,4,4]), torch.tensor([1,1,3,3])))
    # get_loss(torch.tensor([[[k*1000 + j * 100 + i for i in range(1, 21)] for j in range(1,4)] for k in range(1,4)]), None)

    from torch.utils.data import DataLoader
    from utils.datasets import PascalVOCDataset
    from tqdm import tqdm

    from yolov2 import YOLO

    pascal_train_dataset = PascalVOCDataset(image_set="train", root="./pascal_voc")
    pascal_train_dataloader = DataLoader(pascal_train_dataset, batch_size=1, shuffle=True, collate_fn=pascal_train_dataset.collater)

    tepoch = tqdm(pascal_train_dataloader, unit="batch")

    for x_train, y_train in tepoch:
        model = YOLO()
        x = model(x_train)
        get_loss(x, y_train)
        # print(x.shape)
        print(y_train)
        break

    # gts = torch.tensor([[[1,2,3,4,1],
    #         [1,2,3,4,2],
    #         [1,2,3,4,4],
    #         [1,2,3,4,10]]])
    # get_loss(torch.tensor([[[j * 100 + i for i in range(1, 17)] for j in range(1,4)]]), gts, 20)
