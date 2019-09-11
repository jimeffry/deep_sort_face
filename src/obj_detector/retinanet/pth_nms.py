import torch
#from ._ext import nms
import numpy as np

def pth_nms(boxes, scores=None,overlap=0.5, top_k=500):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    if scores is None:
        scores = boxes[:,4]
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep,count
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep,count

def pth_nms_(boxes,scores=None, overlap=0.5,topk=200,mode='Union'):
    pick = []
    count = 0
    if scores is None:
        scores = boxes[:,4]
    if boxes.size()==0:
        return pick,count
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    #s  = np.array(scores)
    #area = np.multiply(x2-x1+1, y2-y1+1)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    w = torch.clamp(w, min=0.0)
    h = torch.clamp(h, min=0.0)
    areas = w * h
    v, idx = scores.sort(0)
    idx = idx[-topk:] 
    #I[-1] have hightest prob score, I[0:-1]->others
    #print('test',y2[idx[0:-1]].size())
    while len(idx)>0:
        xx1 = torch.max(x1[idx[-1]],x1[idx[0:-1]])
        yy1 = torch.max(y1[idx[-1]],y1[idx[0:-1]])
        xx2 = torch.max(x2[idx[-1]],x2[idx[0:-1]])
        yy2 = torch.max(y2[idx[-1]],y2[idx[0:-1]])
        in_w = xx2 - xx1 + 1
        in_h = yy2 - yy1 + 1
        in_w = torch.clamp(in_w,min=0.0)
        in_h = torch.clamp(in_h,min=0.0)
        inter = in_w * in_h
        if mode == 'Min':
            iou = inter / torch.min(areas[idx[-1]], areas[idx[0:-1]])
        else:
            iou = inter / (areas[idx[-1]] + areas[idx[0:-1]] - inter)
        pick.append(idx[-1])
        count +=1
        mask = iou.le(overlap)
        idx = idx[:-1]
        idx = idx[mask]
    return pick,count