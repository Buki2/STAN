# -*- coding: utf-8 -*-

"""
Custom loss function definitions.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class IoULoss(nn.Module):
    """
    Creates a criterion that computes the Intersection over Union (IoU)
    between a segmentation mask and its ground truth.

    Rahman, M.A. and Wang, Y:
    Optimizing Intersection-Over-Union in Deep Neural Networks for
    Image Segmentation. International Symposium on Visual Computing (2016)
    http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        input = F.sigmoid(input)
        intersection = (input * target).sum()
        union = ((input + target) - (input * target)).sum()
        iou = intersection / union
        iou_dual = input.size(0) - iou
        if self.size_average:
            iou_dual = iou_dual / input.size(0)
        return iou_dual
def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h,w=256//stride,256//stride
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords
class GenTargets(nn.Module):
    def __init__(self):
        super().__init__()
        self.strides=[8,16,32]
        self.limit_range=[[-1,64],[64,128],[128,9999]]
        assert len(self.strides)==len(self.limit_range)

    def forward(self,inputs):
        '''
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cnt_logits=inputs[0]
        gt_boxes=inputs[1]
        cnt_targets_all_level=[]
        assert len(self.strides)==len(cnt_logits)
        for level in range(len(cnt_logits)):
            level_out=cnt_logits[level]
            level_targets=self._gen_level_targets(level_out,gt_boxes,self.strides[level],
                                                    self.limit_range[level])
            cnt_targets_all_level.append(level_targets)
            
        return cnt_targets_all_level#torch.cat(cnt_targets_all_level,dim=1)

    def _gen_level_targets(self,out,gt_boxes,stride,limit_range,sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cnt_logits=out
        batch_size=cnt_logits.shape[0]
        coords=coords_fmap2orig(cnt_logits,stride).to(device=gt_boxes.device)

        x=coords[:,0]
        y=coords[:,1]
        
        l_off=x[None,:,None]-gt_boxes[...,0][:,None,None]
        t_off=y[None,:,None]-gt_boxes[...,1][:,None,None]
        r_off=gt_boxes[...,2][:,None,None]-x[None,:,None]
        b_off=gt_boxes[...,3][:,None,None]-y[None,:,None]
        ltrb_off=torch.cat([l_off,t_off,r_off,b_off],dim=-1)

        left_right_min = torch.min(ltrb_off[..., 0], ltrb_off[..., 2]).clamp(min=0)
        left_right_max = torch.max(ltrb_off[..., 0], ltrb_off[..., 2])
        top_bottom_min = torch.min(ltrb_off[..., 1], ltrb_off[..., 3]).clamp(min=0)
        top_bottom_max = torch.max(ltrb_off[..., 1], ltrb_off[..., 3])
        cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt()
        
        return cnt_targets
