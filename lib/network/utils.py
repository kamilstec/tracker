import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def weighted_binary_cross_entropy(output, target, weights=None):
    loss = - weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    return torch.mean(loss)

def sinkhorn(matrix):
    row_len = len(matrix)
    col_len = len(matrix[0])
    if torch.cuda.is_available():
        desired_row_sums = torch.ones((1, row_len), requires_grad=False).cuda()
        desired_col_sums = torch.ones((1, col_len), requires_grad=False).cuda()
    else:
        desired_row_sums = torch.ones((1, row_len), requires_grad=False)
        desired_col_sums = torch.ones((1, col_len), requires_grad=False)
    desired_row_sums[:, -1] = col_len-1
    desired_col_sums[:, -1] = row_len-1

    for _ in range(8):
        actual_row_sum = torch.sum(matrix, axis=1)
        zeroed = matrix.clone()
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                zeroed[i, j] = element * desired_row_sums[0, i] / (actual_row_sum[i])
        actual_col_sum = torch.sum(zeroed, axis=0)

        matrix = zeroed.clone()

        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                zeroed[i, j] = element * desired_col_sums[0, j] / (actual_col_sum[j])
        matrix = zeroed.clone()
    return matrix

def hungarian(output, det_num, tracklet_num):
    cleaned_output = []
    num = 0
    eps = 0.0001
    for i, j in enumerate(tracklet_num):
        matrix = []
        for k in range(j):
            matrix.append([])
            for l in range(det_num[i]):
                matrix[k].append(1 - output[num].cpu().detach().numpy())
                num += 1
        matrix = np.array(matrix)
        (a, b) = matrix.shape
        if a > b:
            padding = ((0, 0), (0, a - b))
        else:
            padding = ((0, b - a), (0, 0))
        matrix = np.pad(matrix, padding, mode='constant', constant_values=eps)
        # hungarian
        row_ind, col_ind = linear_sum_assignment(matrix)
        remove_ind= []
        cnt= 0
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                if element==1:
                    remove_ind.append(cnt)
                cnt += 1
        cnt= 0
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                if i < a and j < b:
                    p1 = row_ind.tolist().index(i)
                    p2 = col_ind.tolist().index(j)
                    # print(p2)
                    if p1 == p2 and cnt not in remove_ind:
                        if torch.cuda.is_available():
                            cleaned_output.append(torch.tensor(1, dtype=float).cuda())
                        else:
                            cleaned_output.append(torch.tensor(1, dtype=float))
                    else:
                        if torch.cuda.is_available():
                            cleaned_output.append(torch.tensor(0, dtype=float).cuda())
                        else:
                            cleaned_output.append(torch.tensor(0, dtype=float))
                cnt += 1
    cleaned_output = torch.stack(cleaned_output)
    return cleaned_output

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou_calc(boxes1, boxes2):
    boxes1= boxes1.reshape(1,4)
    boxes2 = boxes2.reshape(1, 4)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
