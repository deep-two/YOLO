import torch

def bbox_iou(bbox1, bbox2):
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
    match_idx = torch.ones(outputs.shape)
    for gt_idx, gt in enumerate(gts):
        max_iou_idx = -1
        max_iou = -1
        for i, output in enumerate(outputs):
            iou = bbox_iou(gt, output)
            if max_iou < iou:
                max_iou = iou
                max_iou_idx = i
        match_idx[i] = gt_idx
            
    return match_idx

def get_loss(outputs, gts):
    # Todo #########################################
    # 1. reshaping outputs (to x1,y1,x2, y2 of rectangle)
    # 2. reshape gts (to x1,y1,x2, y2 of rectangle + reduction(ex. 32))
    # 3. match outputs and gts (by best iou, same shape of all outputs)
    # 4. calculate loss of conf, box, class
    ################################################

    pred_boxes = None
    pred_conf = None
    pred_class = None

    match_box = match(pred_boxes, gts)
    
    


if __name__ == "__main__":
    print(bbox_iou(torch.tensor([3,3,4,4]), torch.tensor([1,1,3,3])))