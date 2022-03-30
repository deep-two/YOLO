# from cv2 import moments # Error with this line
from socket import INADDR_UNSPEC_GROUP
from cv2 import mean
import torch
from utils.config import cfg

def bbox_iou(inbox1, inbox2):
    iou_out = inbox1.new_zeros(inbox1.size()[0], inbox2.size()[0])

    bbox1_l = inbox1[:, 0]  - inbox1[:, 2] / 2
    bbox1_t = inbox1[:, 1]  - inbox1[:, 3] / 2
    bbox1_r = inbox1[:, 0]  + inbox1[:, 2] / 2
    bbox1_b = inbox1[:, 1]  + inbox1[:, 3] / 2

    bbox2_l = inbox2[:, 0]  - inbox2[:, 2] / 2
    bbox2_t = inbox2[:, 1]  - inbox2[:, 3] / 2
    bbox2_r = inbox2[:, 0]  + inbox2[:, 2] / 2
    bbox2_b = inbox2[:, 1]  + inbox2[:, 3] / 2

    left_x = torch.max(bbox1_l.unsqueeze(-1), bbox2_l)
    top_y = torch.max(bbox1_t.unsqueeze(-1), bbox2_t)
    right_x = torch.min(bbox1_r.unsqueeze(-1), bbox2_r)
    bottom_y = torch.min(bbox1_b.unsqueeze(-1), bbox2_b)

    tt = right_x - left_x
    tt2 = bottom_y - top_y
    t = (right_x - left_x > 0).clone()
    t2 = (bottom_y - top_y > 0).clone()
    tmp = torch.logical_and(right_x - left_x > 0, bottom_y - top_y > 0)
    inds = torch.nonzero(torch.logical_and(right_x - left_x > 0, bottom_y - top_y > 0))
    # if right_x - left_x < 0 or bottom_y - top_y < 0:
    #     return 0

    intersection = (right_x - left_x) * (bottom_y - top_y)
    area1 = (bbox1_r - bbox1_l) * (bbox1_b - bbox1_t)
    area2 = (bbox2_r - bbox2_l) * (bbox2_b - bbox2_t)
    union = area1.unsqueeze(-1) + area2 - intersection

    for x, y in inds:
        iou_out[x,y] = intersection[x,y] / union[x,y]

    return iou_out

def match(outputs, gts):
    result = -outputs.new_zeros(outputs.shape)
    thresh = cfg.TRAIN.MIN_IOU_THRESH

    ious = bbox_iou(outputs[:, :4], gts[:, :4])
    max_ious, max_gt_inds = torch.max(ious, dim=1)

    inds_f = torch.nonzero(max_ious < thresh)
    # max_ious[inds] = 0
    max_gt_inds[inds_f] = 0

    inds_t = torch.nonzero(max_ious >= thresh)
    

    print(len(inds_f))
    print(len(inds_t))

    result[inds_t, 0] = gts[max_gt_inds[inds_t],0]
    result[inds_t, 1] = gts[max_gt_inds[inds_t],1]
    result[inds_t, 2] = gts[max_gt_inds[inds_t],2]
    result[inds_t, 3] = gts[max_gt_inds[inds_t],3]
    # result[:, 4] = 1
    result[:, 4] = max_ious.detach()
    result[inds_t, 5:] = gts[max_gt_inds[inds_t],5:]

    max_gt_inds[inds_t] = 1

    return result, max_gt_inds

def get_loss(outputs, gts, class_num=20, lambda_coordi=5, lambda_nobody = 0.5):
    error_sum = gts.new_zeros(1)
    for batch in range(outputs.shape[0]):
        pred_boxes = outputs[batch].view([-1,5+class_num]).contiguous()

        gt = gts[batch]
        gt_mask = torch.where(gt[:, -1] == -1, False, True)
        gt = gt[gt_mask]
        gt_boxes = gts.new_zeros([len(gt), 5+class_num])
        gt_boxes[:, 4] = 1
        gt_boxes[:, 0] = (gt[:, 0] + gt[:, 2]) / 2 / cfg.TRAIN.FEATURE_STRIDE
        gt_boxes[:, 1] = (gt[:, 1] + gt[:, 3]) / 2 / cfg.TRAIN.FEATURE_STRIDE
        gt_boxes[:, 2] = (gt[:, 2] - gt[:, 0]) / cfg.TRAIN.FEATURE_STRIDE
        gt_boxes[:, 3] = (gt[:, 3] - gt[:, 1]) / cfg.TRAIN.FEATURE_STRIDE
        gt_boxes[:, 5:] = torch.nn.functional.one_hot(gt[:, 4].to(torch.int64), num_classes=class_num)
        
        matched_gt, mask = match(pred_boxes, gt_boxes)

        len_ = mask.size()[0]
        # mask = mask.view((-1,1)).expand((len_, class_num+5)).contiguous()

        coordi_error_x = torch.sum(torch.square(pred_boxes[:,0] - matched_gt[:,0]) * mask)
        coordi_error_y = torch.sum(torch.square(pred_boxes[:,1] - matched_gt[:,1]) * mask)
        coordi_error_w = torch.sum(torch.square(torch.sqrt(pred_boxes[:,2]) - torch.sqrt(matched_gt[:,2])) * mask)
        coordi_error_h = torch.sum(torch.square(torch.sqrt(pred_boxes[:,3]) - torch.sqrt(matched_gt[:,3])) * mask)

        conf_error = torch.sum(torch.square(pred_boxes[:,4] - matched_gt[:,4]) * torch.where(mask == 0, lambda_nobody, 1.))
        class_error = torch.sum(torch.square(pred_boxes[:,5:] - matched_gt[:,5:]) * mask.view((-1,1)).expand((len_, class_num)).contiguous())

        error = lambda_coordi * (coordi_error_x + coordi_error_y + coordi_error_w + coordi_error_h) + conf_error + class_error
        error_sum += error

    return error_sum

    
    


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
