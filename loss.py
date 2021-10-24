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
    iou = -torch.ones(outputs.shape[0])
    thresh = 0.5

    for gt in gts:
        max_iou_idx = -1
        max_iou = -1
        for i, output in enumerate(outputs):
            iou = bbox_iou(gt[:4], output[:4])
            if max_iou < iou and iou > thresh:
                max_iou = iou
                max_iou_idx = i
        result[max_iou_idx] = gt
        iou[max_iou_idx] = max_iou
            
    return result, iou

def get_loss(outputs: torch, gts, lambda_coordi=5, lambda_nobody = 0.5):
    # Todo #########################################
    # 1. reshape gts (same shape with pred_box + reduction(ex. 32))
    # 2. match outputs and gts (by best iou, same shape of all outputs)
    # 3. non-for structure
    ################################################

    class_num = 5
    for batch in range(outputs.shape[0]):
        pred_boxes = outputs[batch].view([-1,5+class_num])

        matched_gt, mask = match(pred_boxes, gts)

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

    return lambda_coordi * coordi_error + conf_error + class_error


    
    


if __name__ == "__main__":
    # print(bbox_iou(torch.tensor([8,8,4,4]), torch.tensor([1,1,3,3])))
    get_loss(torch.tensor([[[k*1000 + j * 100 + i for i in range(1, 21)] for j in range(1,4)] for k in range(1,4)]), None)