from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .darknet import *
from .convlstm import *
from .modulation import *

import argparse
import collections
import logging
import json
import re
import time
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


def st_grid_calculation(st_relevance_score, word_id_st_sent2wordlist, bbox_st_list, word_id_st_sent, st_list_bbox2word, visu_scale, image_scale):
    batch_size = st_relevance_score.shape[0]
    dividend = image_scale // visu_scale
    activation_map = torch.zeros(batch_size, visu_scale, visu_scale, 1).cuda()
    for batch_i in range(batch_size):
        for ii in range(len(st_relevance_score[batch_i])):
            if not st_relevance_score[batch_i][ii] == 0:
                bbox_index = torch.nonzero(st_list_bbox2word[batch_i]==(word_id_st_sent2wordlist[batch_i][ii]+1))
                for jj in bbox_index:
                    x1, y1, x2, y2 = bbox_st_list[batch_i][jj.item()]
                    grid_xl = (x1 // dividend).int().item()
                    grid_xr = min((x2 // dividend + 1).int().item(), visu_scale - 1)
                    grid_yt = (y1 // dividend).int().item()
                    grid_yb = min((y2 // dividend + 1).int().item(), visu_scale - 1)
                    activation_map[batch_i, grid_yt:grid_yb, grid_xl:grid_xr] = st_relevance_score[batch_i][ii].item()
    # grid softmax
    for batch_i in range(batch_size):
        if not len(torch.nonzero(activation_map[batch_i])) == 0:
            tmp = activation_map[batch_i]
            tmp = tmp.reshape(-1, 1)
            tmp = F.softmax(tmp*9, dim=0)
            tmp = tmp.reshape(visu_scale, visu_scale, -1)
            activation_map[batch_i] = tmp
    return activation_map


def generate_coord(batch, height, width):
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord


class cross_attention_head(nn.Module):
    def __init__(self, emb_size=256, tunebert=False, convlstm=False, bert_model='bert-base-uncased', leaky=False, \
        jemb_drop_out=0.1,raw_feature_norm='softmax',NCatt=2,down_sample_ith=2,fpn_n=3,sub_step=2,n_head=3):
        super(cross_attention_head, self).__init__()

        self.down_sample_ith = down_sample_ith
        self.fpn_n = fpn_n
        self.emb_size = emb_size
        self.NCatt = NCatt
        self.sub_step = sub_step
        self.tunebert = tunebert
        self.raw_feature_norm = raw_feature_norm
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        self.convlstm = convlstm
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        ## Text model
        self.textmodel = BertModel.from_pretrained(bert_model)
        self.mapping_visu = ConvBatchNormReLU(256, emb_size, 1, 1, 0, 1, leaky=leaky)
        self.mapping_lang = torch.nn.Sequential(
          nn.Linear(self.textdim, emb_size),
          nn.ReLU(),
          nn.Dropout(jemb_drop_out),
          nn.Linear(emb_size, emb_size),
          nn.ReLU(),)
        self.txt_single_classifier = ConvBatchNormReLU(emb_size*2, 1, 1, 1, 0, 1, leaky=leaky)
        self.softmax = nn.Softmax()
        self.bn =nn.BatchNorm2d(emb_size)
        self.cross_att_modulesdict = nn.ModuleDict()
        output_emb = emb_size
        modules = OrderedDict()
        modules['convmerge0_1x1'] = ConvBatchNormReLU(emb_size * 2, emb_size, 1, 1, 0, 1)
        modules['convmerge1_1x1'] = ConvBatchNormReLU(emb_size * 2, emb_size, 1, 1, 0, 1)
        for i in range(fpn_n-1):
            modules['conv%d_downsample'%i] = torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1),
                nn.MaxPool2d(2,2))
        modules['fcn'] = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb*2, output_emb, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb, 9*5, kernel_size=1))
        modules['fcn_sub'] = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb*2, output_emb, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb, 9*5, kernel_size=1))
        modules['fcn_2sub'] = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb*2, output_emb, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb, 9*5, kernel_size=1))
        kn = 0
        for _ in range(0,self.fpn_n):
            for _ in range(self.sub_step):
                modules['catt%d'%kn] = cross_att_blocked(raw_feature_norm=raw_feature_norm,head=n_head)
                modules['linear%d'%kn] = nn.Linear(1024,emb_size)
                modules['conv%d_1x1'%kn] = ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1)
                modules['conv%d_3x3'%kn] = ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1)
                kn += 1

        self.cross_att_modulesdict.update(modules)


    def forward(self, image, word_id, word_mask, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent, st_list_bbox2word):

        ## Visual Module
        batch_size = image.size(0)
        raw_fvisu = self.visumodel(image)
        if self.convlstm:
            raw_fvisu = raw_fvisu[1]
        else:
            raw_fvisu_8x8 = raw_fvisu[0]
            raw_fvisu_16x16 = raw_fvisu[1]
            raw_fvisu = raw_fvisu[2]

        ## Language Module for scene text
        all_encoder_layers_st_sent, _ = self.textmodel(word_id_st_sent, token_type_ids=None, attention_mask=word_mask_st_sent)
        raw_flang_st_sent = (all_encoder_layers_st_sent[-1][:, 0, :] + all_encoder_layers_st_sent[-2][:, 0, :] +
                              all_encoder_layers_st_sent[-3][:, 0, :] + all_encoder_layers_st_sent[-4][:, 0, :]) / 4
        raw_fword_st_sent = (all_encoder_layers_st_sent[-1] + all_encoder_layers_st_sent[-2] +
                              all_encoder_layers_st_sent[-3] + all_encoder_layers_st_sent[-4]) / 4
        if not self.tunebert:
            hidden_st_sent = raw_flang_st_sent.detach()
            raw_fword_st_sent = raw_fword_st_sent.detach()

        ## Language Module for expression
        all_encoder_layers, _ = self.textmodel(word_id, \
            token_type_ids=None, attention_mask=word_mask)
        ## Sentence feature at the first position [cls]
        raw_flang = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:]\
             + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
        raw_fword = (all_encoder_layers[-1] + all_encoder_layers[-2]\
             + all_encoder_layers[-3] + all_encoder_layers[-4])/4
        if not self.tunebert:
            ## fix bert during training
            # raw_flang = raw_flang.detach()
            hidden = raw_flang.detach()
            raw_fword = raw_fword.detach()

        ## Correlatd Text Extraction & Correlated Region Activation
        mask_word_att = torch.zeros_like(raw_fword).cuda()
        mask_st_att = torch.zeros_like(raw_fword_st_sent).cuda()
        for ii in range(batch_size):
            mask_word_att[ii, 1:len(torch.nonzero(word_mask[ii])) - 1, :] = 1
            mask_st_att[ii, 1:len(torch.nonzero(word_id_st_sent[ii])) - 1, :] = 1
        raw_fword_attn = raw_fword * mask_word_att
        raw_fword_st_sent = raw_fword_st_sent * mask_st_att
        st_relevance_score = torch.zeros(batch_size, mask_st_att.size(1), mask_word_att.size(1)).cuda()

        THRES_PHI = 0.50
        for ii in range(batch_size):
            st_relevance_score[ii] = F.cosine_similarity(raw_fword_st_sent[ii].unsqueeze(1), raw_fword_attn[ii], dim=-1)
        st_relevance_score = torch.max(st_relevance_score, dim=2, keepdim=True).values
        st_relevance_score = torch.where(st_relevance_score < THRES_PHI, torch.zeros_like(st_relevance_score), st_relevance_score)

        weighted_st_feature_8x8 = st_grid_calculation(st_relevance_score, word_st_position, bbox_st_list, word_id_st_sent, st_list_bbox2word, raw_fvisu_8x8.size(2), image.size(2))
        raw_fvisu_8x8 = raw_fvisu_8x8.permute(0,2,3,1).contiguous() * weighted_st_feature_8x8 + raw_fvisu_8x8.permute(0,2,3,1).contiguous()
        raw_fvisu_8x8 = raw_fvisu_8x8.permute(0,3,1,2).contiguous()

        weighted_st_feature_16x16 = st_grid_calculation(st_relevance_score, word_st_position, bbox_st_list, word_id_st_sent, st_list_bbox2word, raw_fvisu_16x16.size(2), image.size(2))
        raw_fvisu_16x16 = raw_fvisu_16x16.permute(0,2,3,1).contiguous() * weighted_st_feature_16x16 + raw_fvisu_16x16.permute(0,2,3,1).contiguous()
        raw_fvisu_16x16 = raw_fvisu_16x16.permute(0,3,1,2).contiguous()

        weighted_st_feature_32x32 = st_grid_calculation(st_relevance_score, word_st_position, bbox_st_list, word_id_st_sent, st_list_bbox2word, raw_fvisu.size(2),image.size(2))
        raw_fvisu = raw_fvisu.permute(0,2,3,1).contiguous() * weighted_st_feature_32x32 + raw_fvisu.permute(0,2,3,1).contiguous()
        raw_fvisu = raw_fvisu.permute(0,3,1,2).contiguous()

        ## Language Module - mapping language feature
        fword = Variable(torch.zeros(raw_fword.shape[0], raw_fword.shape[1], self.emb_size).cuda())
        for ii in range(raw_fword.shape[0]):
            ntoken = (word_mask[ii] != 0).sum()
            fword[ii, :ntoken, :] = F.normalize(self.mapping_lang(raw_fword[ii, :ntoken, :]), p=2, dim=1)
        raw_fword = fword
        global_raw_fword = raw_fword.mean(1)

        ## Visual Module - mapping visual feature & decomposition
        fvisu = self.mapping_visu(raw_fvisu)
        raw_fvisu = F.normalize(fvisu, p=2, dim=1)  # 32x32
        raw_fvisu_16x16 = F.normalize(raw_fvisu_16x16, p=2, dim=1)  # 16x16
        raw_fvisu_8x8 = raw_fvisu_8x8.view(batch_size, raw_fvisu_8x8.size(1), -1).transpose(1,2).contiguous()
        raw_fvisu_8x8 = F.max_pool1d(raw_fvisu_8x8, 2).transpose(1, 2).contiguous().view(batch_size, -1, 8, 8)
        raw_fvisu_8x8 = F.normalize(raw_fvisu_8x8, p=2, dim=1)  # 8x8

        map_fvisu = raw_fvisu.view(batch_size, raw_fvisu.size(1), -1)
        map_fvisu_orig = torch.transpose(map_fvisu, 1, 2).contiguous()
        map_fvisu_16x16 = raw_fvisu_16x16.view(batch_size, raw_fvisu_16x16.size(1), -1)
        map_fvisu_16x16 = torch.transpose(map_fvisu_16x16, 1, 2).contiguous()
        map_fvisu_8x8 = raw_fvisu_8x8.view(batch_size, raw_fvisu_8x8.size(1), -1)
        map_fvisu_8x8 = torch.transpose(map_fvisu_8x8, 1, 2).contiguous()
        map_fvisu_orig_co = map_fvisu_orig.clone()

        ## Visual Module - location feature
        coord = generate_coord(batch_size, raw_fvisu.size(2), raw_fvisu.size(3))
        coord_16x16 = generate_coord(batch_size, raw_fvisu_16x16.size(2), raw_fvisu_16x16.size(3))
        coord_8x8 = generate_coord(batch_size, raw_fvisu_8x8.size(2), raw_fvisu_8x8.size(3))

        map_coord = coord.view(batch_size, coord.size(1), -1)
        map_coord = torch.transpose(map_coord, 1, 2).contiguous()
        map_coord_16x16 = coord_16x16.view(batch_size, coord_16x16.size(1), -1)
        map_coord_16x16 = torch.transpose(map_coord_16x16, 1, 2).contiguous()
        map_coord_8x8 = coord_8x8.view(batch_size, coord_8x8.size(1), -1)
        map_coord_8x8 = torch.transpose(map_coord_8x8, 1, 2).contiguous()

        ## Initialization for bottom-up and bidirectional fusion
        make_f = []
        make_target_visu = []
        make_target_txt = []
        out_feat = []
        cosine_weights = []
        contrast_visu = 0
        contrast_txt = 0
        cosine_txt_word, cosine_txt_visu = None, None
        map_fvisu_add = map_fvisu_orig
        map_coord_add = map_coord
        raw_fvisu_add = raw_fvisu
        out_visu = 0
        merge_t = 0
        att_n = 0

        for ff in range(self.fpn_n):  # for multi-scale visual features
            for n in range(self.sub_step):  # for multi-step alignment
                if ff != 0 or n != 0:
                    out_visu = merge_f.view(batch_size, raw_fvisu.size(1), -1)
                    out_visu = torch.transpose(out_visu, 1, 2).contiguous()
                out_visu, out_txt, cosine_txt_region, cosine_visu_region, cosine_txt_word, cosine_txt_visu = self.cross_att_modulesdict['catt%d'%att_n](out_visu+map_fvisu_add, merge_t+raw_fword, map_coord_add,cosine_txt_word, cosine_txt_visu, word_mask)

                out_visu = out_visu + global_raw_fword.unsqueeze(1)
                out_visu = torch.transpose(out_visu, 1, 2).contiguous()
                out_visu = out_visu.view(batch_size,raw_fvisu_add.size(1),raw_fvisu_add.size(2),raw_fvisu_add.size(3))
                merge_f = torch.cat([raw_fvisu_add+contrast_visu,out_visu],dim=1)
                merge_f = self.cross_att_modulesdict['conv%d_1x1'%att_n](merge_f)
                merge_f = self.cross_att_modulesdict['conv%d_3x3'%att_n](merge_f)
                merge_t = torch.cat([raw_fword+contrast_txt,out_txt],dim=-1)
                merge_t = self.cross_att_modulesdict['linear%d'%att_n](merge_t)
                make_target_visu.extend(cosine_visu_region)
                make_target_txt.extend(cosine_txt_region)
                make_f.append(merge_f)
                att_n += 1
            if ff == 0:
                max_feature_32x32 = torch.stack(make_f[:self.sub_step],-1).sum(-1)
                merge_f = self.cross_att_modulesdict['conv0_downsample'](max_feature_32x32)
                raw_fvisu_add = raw_fvisu_16x16
                map_fvisu_add = map_fvisu_16x16
                map_coord_add = map_coord_16x16

            elif ff == 1:
                max_feature_16x16 = torch.stack(make_f[self.sub_step:self.sub_step*(ff+1)],-1).sum(-1)
                merge_f = self.cross_att_modulesdict['conv1_downsample'](max_feature_16x16)
                raw_fvisu_add = raw_fvisu_8x8
                map_fvisu_add = map_fvisu_8x8
                map_coord_add = map_coord_8x8
            if ff == self.fpn_n - 1 and n == self.sub_step - 1:
                max_feature_8x8 = torch.stack(make_f[self.sub_step*ff:],-1).sum(-1)
                upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)
                fpn_region_16x16 = upsampling1(max_feature_8x8)
                fpn_region_16x16 = torch.cat([max_feature_16x16, fpn_region_16x16], dim=1)
                fpn_region_16x16 = self.cross_att_modulesdict['convmerge0_1x1'](fpn_region_16x16)
                fpn_region_32x32 = upsampling1(fpn_region_16x16)
                fpn_region_32x32 = torch.cat([max_feature_32x32, fpn_region_32x32], dim=1)
                fpn_region_32x32 = self.cross_att_modulesdict['convmerge1_1x1'](fpn_region_32x32)

                out_region_32x32 = self.cross_att_modulesdict['fcn'](torch.cat([fpn_region_32x32, raw_fvisu], dim=1))
                out_region_16x16 = self.cross_att_modulesdict['fcn_sub'](torch.cat([fpn_region_16x16, raw_fvisu_16x16], dim=1))
                out_region_8x8 = self.cross_att_modulesdict['fcn_2sub'](torch.cat([max_feature_8x8, raw_fvisu_8x8], dim=1))

        single_conf = self.txt_single_classifier(torch.cat([max_feature_32x32,raw_fvisu],dim=1)).view(batch_size,map_fvisu_orig_co.size(1))
        single_conf_16 = self.txt_single_classifier(torch.cat([max_feature_16x16,raw_fvisu_16x16],dim=1)).view(batch_size,16*16)
        single_conf_8 = self.txt_single_classifier(torch.cat([max_feature_8x8,raw_fvisu_8x8],dim=1)).view(batch_size,8*8)
        out_feat.extend([out_region_32x32,out_region_16x16,out_region_8x8])
        cosine_weights.extend([make_target_visu,make_target_txt,word_mask,single_conf,single_conf_16,single_conf_8])

        return out_feat, cosine_weights


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, raw_feature_norm, smooth=9, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)

    if weight is not None:
      attn = attn + weight

    attn_out = attn.clone()

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attn_out
