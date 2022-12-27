import os
import sys
import argparse
import time
import random
import json
import math
from distutils.version import LooseVersion
import scipy.misc
import logging
import datetime
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import ResizeImage, ResizeAnnotation

from dataset.data_loader import *
from model.grounding_model import *
from model.loss import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume, load_resume_optimizer

def main():
    parser = argparse.ArgumentParser(description='Dataloader test')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=16, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--power', default=0, type=float, help='lr poly power; 0 indicates step decay by half')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--size', default=256, type=int, help='image size')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='reftext', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=500, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--test_set', default='test', type=str, help='test/subtest_street/subtest_shelf/...')
    parser.add_argument('--ncatt', default=9, type=int, help='ncatt')
    parser.add_argument('--mstage', dest='mstage', default=False, action='store_true', help='if mstage')
    parser.add_argument('--mstack', dest='mstack', default=False, action='store_true', help='if mstack')
    parser.add_argument('--w_div', default=3, type=float, help='weight of the diverge loss')
    parser.add_argument('--down_sample_ith', default=2, type=int, help='down_sample of ith')
    parser.add_argument('--n_head', default=4, type=int, help='number of decomposing head')
    parser.add_argument('--fpn_n', default=3, type=int, help='number of fpn feature')
    parser.add_argument('--sub_step_n', default=3, type=int, help='number of steps in each scale')
    parser.add_argument('--decomp_feat_n', default=4, type=int, help='number of decomposition feature')
    parser.add_argument('--fusion', default='prod', type=str, help='prod/cat')
    parser.add_argument('--tunebert', dest='tunebert', default=False, action='store_true', help='if tunebert')
    parser.add_argument('--large', dest='large', default=False, action='store_true', help='if large mode: fpn16, convlstm out, size 512')
    parser.add_argument('--raw_feature_norm', default="no_norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    global args, anchors_full
    args = parser.parse_args()
    if args.large:
        args.gsize = 16
        args.size = 512
    else:
        args.gsize = 8
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
    ## following anchor sizes calculated by kmeans under args.anchor_imsize=416
    if args.dataset=='referit':
        anchors = '30,36,  78,46,  48,86,  149,79,  82,148,  331,93,  156,207,  381,163,  329,285'
    elif args.dataset=='flickr':
        anchors = '29,26,  55,58,  137,71,  82,121,  124,205,  204,132,  209,263,  369,169,  352,294'
    elif args.dataset == 'reftext':
        anchors = '26,45, 30,104, 47,78, 48,180, 73,44, 74,111, 78,214, 129,116, 130,219'
    else:
        anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

    ## save logs
    if args.savename=='default':
        args.savename = 'STAN_%s_batch%d'%(args.dataset,args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    if args.test:
        ## note certain dataset does not have 'test' set:
        ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
        test_dataset = ReferDataset(data_root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            testmode=True,
                            split=args.test_set,
                            imsize = args.size,
                            transform=input_transform,
                            max_query_len=args.time)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

    else:
        train_dataset = ReferDataset(data_root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            split='train',
                            imsize = args.size,
                            transform=input_transform,
                            max_query_len=args.time,
                            augment=True)
        val_dataset = ReferDataset(data_root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            split='val',
                            imsize = args.size,
                            transform=input_transform,
                            max_query_len=args.time)
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                pin_memory=True, drop_last=True, num_workers=args.workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, drop_last=True, num_workers=args.workers)
    
    ## Model
    model = cross_attention_head(emb_size=args.emb_size, tunebert=args.tunebert, convlstm=args.large, \
        bert_model=args.bert_model,raw_feature_norm=args.raw_feature_norm, NCatt=args.ncatt,down_sample_ith=args.down_sample_ith,fpn_n=args.fpn_n,sub_step=args.sub_step_n,n_head=args.n_head)
    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        model=load_pretrain(model,args,logging)
    if args.resume:
        model=load_resume(model,args,logging)

    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    if args.tunebert:
        visu_param = model.module.visumodel.parameters()
        text_param = model.module.textmodel.parameters()
        rest_param = [param for param in model.parameters() if ((param not in visu_param) and (param not in text_param))]
        visu_param = list(model.module.visumodel.parameters())
        text_param = list(model.module.textmodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in text_param])
        sum_fusion = sum([param.nelement() for param in rest_param])
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)
    else:
        visu_param = model.module.visumodel.parameters()
        rest_param = [param for param in model.parameters() if param not in visu_param]
        visu_param = list(model.module.visumodel.parameters())
        sum_visu = sum([param.nelement() for param in visu_param])
        sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
        sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    ## optimizer; rmsprop default
    if args.tunebert:
        optimizer = torch.optim.RMSprop([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.},
                {'params': text_param, 'lr': args.lr/10.}], lr=args.lr, weight_decay=0.0005)
    else:
        optimizer = torch.optim.RMSprop([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.}],lr=args.lr, weight_decay=0.0005)
    if args.resume:
        optimizer = load_resume_optimizer(optimizer,args,logging)

    ## training and testing
    best_accu = -float('Inf')
    if args.test:
        _ = test_epoch(test_loader, model)
    else:
        for epoch in range(args.nb_epoch):
            adjust_learning_rate(args, optimizer, epoch)
            train_epoch(train_loader, model, optimizer, epoch)
            accu_new = validate_epoch(val_loader, model)
            ## remember best accu and save checkpoint
            is_best = accu_new >= best_accu
            best_accu = max(accu_new, best_accu)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': accu_new,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, filename=args.savename)
        print('\nBest Accu: %f\n'%best_accu)
        logging.info('\nBest Accu: %f\n'%best_accu)


def train_epoch(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    div_losses = AverageMeter()
    acc_8 = AverageMeter()
    acc_16 = AverageMeter()
    acc_32 = AverageMeter()
    acc_8_iou9 = AverageMeter()
    acc_16_iou9 = AverageMeter()
    acc_32_iou9 = AverageMeter()

    model.train()
    end = time.time()
    for batch_idx, (imgs, word_id, word_mask, bbox, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent, st_list_bbox2word) in enumerate(train_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        word_st_position = word_st_position.cuda()
        bbox_st_list = bbox_st_list.cuda()
        word_id_st_sent = word_id_st_sent.cuda()
        word_mask_st_sent = word_mask_st_sent.cuda()
        st_list_bbox2word = st_list_bbox2word.cuda()

        word_st_position = Variable(word_st_position)
        bbox_st_list = Variable(bbox_st_list)
        word_id_st_sent = Variable(word_id_st_sent)
        word_mask_st_sent = Variable(word_mask_st_sent)
        st_list_bbox2word = Variable(st_list_bbox2word)

        pred_anchor_list, attnscore_list = model(image, word_id, word_mask, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent, st_list_bbox2word)
        loss = 0.
        div_loss = 0.
        pred_collect = []
        scale_on_loss = [0.8,1.5,3]
        for nn,pred_anchor in enumerate(pred_anchor_list):
            ## convert gt box to center+offset format
            gt_param, gi, gj, best_n_list = build_target(bbox, pred_anchor, anchors_full, args)
            pred_collect.append([best_n_list,gi,gj])
            ## flatten anchor dim at each scale
            pred_anchor = pred_anchor.view(   \
                    pred_anchor.size(0),9,5,pred_anchor.size(2),pred_anchor.size(3))
            ## loss
            loss0,txt_loss = yolo_loss(pred_anchor, gt_param, gi, gj, best_n_list, attnscore_list,args=args)
            loss += loss0*scale_on_loss[nn]
            
            
        pred_anchor = pred_anchor_list[1].view(pred_anchor_list[1].size(0),\
            9,5,pred_anchor_list[1].size(2),pred_anchor_list[1].size(3))
        pred_anchor_fine = pred_anchor_list[0].view(pred_anchor_list[0].size(0),\
            9,5,pred_anchor_list[0].size(2),pred_anchor_list[0].size(3))
        pred_anchor_8x8 = pred_anchor_list[2].view(pred_anchor_list[2].size(0),\
            9,5,pred_anchor_list[2].size(2),pred_anchor_list[2].size(3))
        ## diversity regularization
        div_loss += txt_loss
        div_losses.update(div_loss.item(), imgs.size(0))
        loss += div_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))

        ## training offset eval: if correct with gt center loc
        ## convert offset pred to boxes
        if batch_idx % args.print_freq == 0:
            pred_coord_8x8 = torch.zeros(args.batch_size,4)
            grid, grid_size = args.size//(args.gsize*4), args.gsize*4
            anchor_idxs = range(9)
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]
            for ii in range(args.batch_size):
                pred_coord_8x8[ii,0] = F.sigmoid(pred_anchor_8x8[ii, pred_collect[2][0][ii], 0, pred_collect[2][2][ii], pred_collect[2][1][ii]]) + pred_collect[2][1][ii].float()
                pred_coord_8x8[ii,1] = F.sigmoid(pred_anchor_8x8[ii, pred_collect[2][0][ii], 1, pred_collect[2][2][ii], pred_collect[2][1][ii]]) + pred_collect[2][2][ii].float()
                pred_coord_8x8[ii,2] = torch.exp(pred_anchor_8x8[ii, pred_collect[2][0][ii], 2, pred_collect[2][2][ii], pred_collect[2][1][ii]]) * scaled_anchors[pred_collect[2][0][ii]][0]
                pred_coord_8x8[ii,3] = torch.exp(pred_anchor_8x8[ii, pred_collect[2][0][ii], 3, pred_collect[2][2][ii], pred_collect[2][1][ii]]) * scaled_anchors[pred_collect[2][0][ii]][1]
                pred_coord_8x8[ii,:] = pred_coord_8x8[ii,:] * grid_size
            pred_coord_8x8 = xywh2xyxy(pred_coord_8x8)

            pred_coord = torch.zeros(args.batch_size,4)
            grid, grid_size = args.size//(args.gsize*2), args.gsize*2
            anchor_idxs = range(9)
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]
            for ii in range(args.batch_size):
                pred_coord[ii,0] = F.sigmoid(pred_anchor[ii, pred_collect[1][0][ii], 0, pred_collect[1][2][ii], pred_collect[1][1][ii]]) + pred_collect[1][1][ii].float()
                pred_coord[ii,1] = F.sigmoid(pred_anchor[ii, pred_collect[1][0][ii], 1, pred_collect[1][2][ii], pred_collect[1][1][ii]]) + pred_collect[1][2][ii].float()
                pred_coord[ii,2] = torch.exp(pred_anchor[ii, pred_collect[1][0][ii], 2, pred_collect[1][2][ii], pred_collect[1][1][ii]]) * scaled_anchors[pred_collect[1][0][ii]][0]
                pred_coord[ii,3] = torch.exp(pred_anchor[ii, pred_collect[1][0][ii], 3, pred_collect[1][2][ii], pred_collect[1][1][ii]]) * scaled_anchors[pred_collect[1][0][ii]][1]
                pred_coord[ii,:] = pred_coord[ii,:] * grid_size
            pred_coord = xywh2xyxy(pred_coord)

            ## training offset eval: if correct with gt center loc
            ## convert offset pred to boxes
            pred_coord_fine = torch.zeros(args.batch_size,4)
            grid, grid_size = args.size//(args.gsize), args.gsize
            anchor_idxs = range(9)
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]
            for ii in range(args.batch_size):
                pred_coord_fine[ii,0] = F.sigmoid(pred_anchor_fine[ii, pred_collect[0][0][ii], 0, pred_collect[0][2][ii], pred_collect[0][1][ii]]) + pred_collect[0][1][ii].float()
                pred_coord_fine[ii,1] = F.sigmoid(pred_anchor_fine[ii, pred_collect[0][0][ii], 1, pred_collect[0][2][ii], pred_collect[0][1][ii]]) + pred_collect[0][2][ii].float()
                pred_coord_fine[ii,2] = torch.exp(pred_anchor_fine[ii, pred_collect[0][0][ii], 2, pred_collect[0][2][ii], pred_collect[0][1][ii]]) * scaled_anchors[pred_collect[0][0][ii]][0]
                pred_coord_fine[ii,3] = torch.exp(pred_anchor_fine[ii, pred_collect[0][0][ii], 3, pred_collect[0][2][ii], pred_collect[0][1][ii]]) * scaled_anchors[pred_collect[0][0][ii]][1]
                pred_coord_fine[ii,:] = pred_coord_fine[ii,:] * grid_size
            pred_coord_fine = xywh2xyxy(pred_coord_fine)
            ## box iou
            target_bbox = bbox
            
            iou_8 = bbox_iou(pred_coord_8x8, target_bbox.data.cpu(), x1y1x2y2=True)
            iou_16 = bbox_iou(pred_coord, target_bbox.data.cpu(), x1y1x2y2=True)
            iou_32 = bbox_iou(pred_coord_fine, target_bbox.data.cpu(), x1y1x2y2=True)

            accu_8 = np.sum(np.array((iou_8.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
            accu_16 = np.sum(np.array((iou_16.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
            accu_32 = np.sum(np.array((iou_32.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
            accu_8_iou9 = np.sum(np.array((iou_8.data.cpu().numpy()>0.75),dtype=float))/args.batch_size
            accu_16_iou9 = np.sum(np.array((iou_16.data.cpu().numpy()>0.75),dtype=float))/args.batch_size
            accu_32_iou9 = np.sum(np.array((iou_32.data.cpu().numpy()>0.75),dtype=float))/args.batch_size
            ## metrics
            acc_8.update(accu_8, imgs.size(0))
            acc_16.update(accu_16, imgs.size(0))
            acc_32.update(accu_32, imgs.size(0))
            acc_8_iou9.update(accu_8_iou9, imgs.size(0))
            acc_16_iou9.update(accu_16_iou9, imgs.size(0))
            acc_32_iou9.update(accu_32_iou9, imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % args.print_freq == 0:
                print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    'Div Loss {div.val:.4f} ({div.avg:.4f})\t' \
                    'Accu_8x {acc_8.val:.4f} ({acc_8.avg:.4f})\t' \
                    'Accu_16x {acc_16.val:.4f} ({acc_16.avg:.4f})\t' \
                    'Accu_32x {acc_32.val:.4f} ({acc_32.avg:.4f})\t' \
                    'Accu_iou9_8x {acc_iou9_8.val:.4f} ({acc_iou9_8.avg:.4f})\t' \
                    'Accu_iou9_16x {acc_iou9_16.val:.4f} ({acc_iou9_16.avg:.4f})\t' \
                    'Accu_iou9_32x {acc_iou9_32.val:.4f} ({acc_iou9_32.avg:.4f})\t' \
                    .format( \
                        epoch, batch_idx, len(train_loader), batch_time=batch_time, \
                        data_time=data_time, loss=losses, div=div_losses, \
                        acc_8=acc_8, acc_16=acc_16, acc_32=acc_32, acc_iou9_8=acc_8_iou9, acc_iou9_16=acc_16_iou9, acc_iou9_32=acc_32_iou9)
                print(print_str)
                logging.info(print_str)


def validate_epoch(val_loader, model, mode='val'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc_8 = AverageMeter()
    acc_16 = AverageMeter()
    acc_32 = AverageMeter()

    model.eval()
    end = time.time()
    print(datetime.datetime.now())
    
    for batch_idx, (imgs, word_id, word_mask, bbox, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent, st_list_bbox2word) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        word_st_position = word_st_position.cuda()
        bbox_st_list = bbox_st_list.cuda()
        word_id_st_sent = word_id_st_sent.cuda()
        word_mask_st_sent = word_mask_st_sent.cuda()
        st_list_bbox2word = st_list_bbox2word.cuda()

        word_st_position = Variable(word_st_position)
        bbox_st_list = Variable(bbox_st_list)
        word_id_st_sent = Variable(word_id_st_sent)
        word_mask_st_sent = Variable(word_mask_st_sent)
        st_list_bbox2word = Variable(st_list_bbox2word)

        with torch.no_grad():
            pred_anchor_list, attnscore_list = model(image, word_id, word_mask, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent, st_list_bbox2word)
        pred_collect = []
        for head in pred_anchor_list:
            pred_anchor = head
            gt_param, target_gi, target_gj, best_n_list = build_target(bbox, pred_anchor, anchors_full, args)
            pred_anchor = pred_anchor.view(   \
                        pred_anchor.size(0),9,5,pred_anchor.size(2),pred_anchor.size(3))
            pred_collect.append([best_n_list,target_gi,target_gj,pred_anchor])
        
        ## eval: convert center+offset to box prediction
        ## calculate at rescaled image during validation for speed-up
        pred_conf_8x8 = pred_collect[2][3][:,:,4,:,:].contiguous().view(args.batch_size,-1)
        max_conf, max_loc = torch.max(pred_conf_8x8, dim=1)

        pred_bbox_8x8 = torch.zeros(args.batch_size,4)
        pred_gi, pred_gj, pred_best_n = [],[],[]
        
        grid, grid_size = args.size//(args.gsize*4), args.gsize*4
        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]
        pred_conf_8x8 = pred_collect[2][3][:,:,4,:,:].data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()
        for ii in range(args.batch_size):
            (best_n, gj, gi) = np.where(pred_conf_8x8[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n)

            pred_bbox_8x8[ii,0] = F.sigmoid(pred_collect[2][3][ii, best_n, 0, gj, gi]) + gi
            pred_bbox_8x8[ii,1] = F.sigmoid(pred_collect[2][3][ii, best_n, 1, gj, gi]) + gj
            pred_bbox_8x8[ii,2] = torch.exp(pred_collect[2][3][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox_8x8[ii,3] = torch.exp(pred_collect[2][3][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox_8x8[ii,:] = pred_bbox_8x8[ii,:] * grid_size
        pred_bbox_8x8 = xywh2xyxy(pred_bbox_8x8)

        pred_conf = pred_collect[1][3][:,:,4,:,:].contiguous().view(args.batch_size,-1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(args.batch_size,4)
        pred_gi, pred_gj, pred_best_n = [],[],[]
        
        grid, grid_size = args.size//(args.gsize*2), args.gsize*2
        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]
        pred_conf = pred_collect[1][3][:,:,4,:,:].data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()
        for ii in range(args.batch_size):
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n)

            pred_bbox[ii,0] = F.sigmoid(pred_collect[1][3][ii, best_n, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_collect[1][3][ii, best_n, 1, gj, gi]) + gj
            pred_bbox[ii,2] = torch.exp(pred_collect[1][3][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox[ii,3] = torch.exp(pred_collect[1][3][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)

        ## fine 
        pred_conf_fine = pred_collect[0][3][:,:,4,:,:].contiguous().view(args.batch_size,-1)
        max_conf, max_loc = torch.max(pred_conf_fine, dim=1)

        pred_bbox_fine = torch.zeros(args.batch_size,4)
        pred_gi, pred_gj, pred_best_n = [],[],[]
        grid, grid_size = args.size//args.gsize, args.gsize

        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]
        #for ii in range(args.batch_size):
        pred_conf_fine = pred_collect[0][3][:,:,4,:,:].data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()
        #for ii in range(args.batch_size):
        
        for ii in range(args.batch_size):
            (best_n, gj, gi) = np.where(pred_conf_fine[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            pred_best_n.append(best_n)
            pred_bbox_fine[ii,0] = F.sigmoid(pred_collect[0][3][ii, best_n, 0, gj, gi]) + gi
            pred_bbox_fine[ii,1] = F.sigmoid(pred_collect[0][3][ii, best_n, 1, gj, gi]) + gj
            pred_bbox_fine[ii,2] = torch.exp(pred_collect[0][3][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            pred_bbox_fine[ii,3] = torch.exp(pred_collect[0][3][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox_fine[ii,:] = pred_bbox_fine[ii,:] * grid_size
        pred_bbox_fine = xywh2xyxy(pred_bbox_fine)
        target_bbox = bbox

        ## metrics
        iou_8 = bbox_iou(pred_bbox_8x8, target_bbox.data.cpu(), x1y1x2y2=True)
        iou_16 = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        iou_32 = bbox_iou(pred_bbox_fine, target_bbox.data.cpu(), x1y1x2y2=True)
        accu_8 = np.sum(np.array((iou_8.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        accu_16 = np.sum(np.array((iou_16.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        accu_32 = np.sum(np.array((iou_32.data.cpu().numpy()>0.5),dtype=float))/args.batch_size

        acc_8.update(accu_8, imgs.size(0))
        acc_16.update(accu_16, imgs.size(0))
        acc_32.update(accu_32, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu_8x {acc_8.val:.4f} ({acc_8.avg:.4f})\t' \
                'Accu_16x {acc_16.val:.4f} ({acc_16.avg:.4f})\t' \
                'Accu_32x {acc_32.val:.4f} ({acc_32.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, acc_8=acc_8, acc_16=acc_16, acc_32=acc_32)
            print(print_str)
            logging.info(print_str)
    acc_max = max([acc_8.avg, acc_16.avg, acc_32.avg])
    print("Accu: %f, accu_8x8: %f, accu_16x16: %f, accu_32x32: %f" % (acc_max, acc_8.avg, acc_16.avg, acc_32.avg))
    logging.info("Accu: %f, accu_8x8: %f, accu_16x16: %f, accu_32x32: %f" % (acc_max, acc_8.avg, acc_16.avg, acc_32.avg))
    return acc_max


def test_epoch(val_loader, model, mode='test'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc_center = AverageMeter()
    acc_32 = AverageMeter()
    acc_16 = AverageMeter()
    acc_8 = AverageMeter()
    model.eval()
    end = time.time()
    for batch_idx, (imgs, word_id, word_mask, bbox, ratio, dw, dh, im_id, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent, st_list_bbox2word) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id[1].cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        word_st_position = word_st_position.cuda()
        bbox_st_list = bbox_st_list.cuda()
        word_id_st_sent = word_id_st_sent.cuda()
        word_mask_st_sent = word_mask_st_sent.cuda()
        st_list_bbox2word = st_list_bbox2word.cuda()

        word_st_position = Variable(word_st_position)
        bbox_st_list = Variable(bbox_st_list)
        word_id_st_sent = Variable(word_id_st_sent)
        word_mask_st_sent = Variable(word_mask_st_sent)
        st_list_bbox2word = Variable(st_list_bbox2word)

        conf_collect = []
        with torch.no_grad():
            pred_anchor_list, attnscore_list = model(image, word_id, word_mask, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent, st_list_bbox2word)
        pred_collect = []
        for head in pred_anchor_list:
            pred_anchor = head
            gt_param, target_gi, target_gj, best_n_list = build_target(bbox, pred_anchor, anchors_full, args)
            pred_anchor = pred_anchor.view(   \
                        pred_anchor.size(0),9,5,pred_anchor.size(2),pred_anchor.size(3))
            pred_collect.append([best_n_list,target_gi,target_gj,pred_anchor])
        
        ## eval: convert center+offset to box prediction
        ## calculate at rescaled image during validation for speed-up
        pred_conf_8x8 = pred_collect[2][3][:,:,4,:,:].contiguous().view(1,-1)
        max_conf, max_loc = torch.max(pred_conf_8x8, dim=1)

        pred_bbox_8x8 = torch.zeros(1,4)
        pred_gi, pred_gj, pred_best_n = [],[],[]
        
        grid, grid_size = args.size//(args.gsize*4), args.gsize*4
       
        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]
        pred_conf_8x8 = pred_collect[2][3][:,:,4,:,:].data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()
        conf_collect.append(max_conf_ii[0])
        (best_n, gj, gi) = np.where(pred_conf_8x8[0,:,:,:] == max_conf_ii[0])
        best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
        pred_gi.append(gi)
        pred_gj.append(gj)
        pred_best_n.append(best_n)
        pred_bbox_8x8[0,0] = F.sigmoid(pred_collect[2][3][0, best_n, 0, gj, gi]) + gi
        pred_bbox_8x8[0,1] = F.sigmoid(pred_collect[2][3][0, best_n, 1, gj, gi]) + gj
        pred_bbox_8x8[0,2] = torch.exp(pred_collect[2][3][0, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox_8x8[0,3] = torch.exp(pred_collect[2][3][0, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
        pred_bbox_8x8[0,:] = pred_bbox_8x8[0,:] * grid_size
        pred_bbox_8x8 = xywh2xyxy(pred_bbox_8x8)

        pred_conf = pred_collect[1][3][:,:,4,:,:].contiguous().view(1,-1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(1,4)
        pred_gi, pred_gj, pred_best_n = [],[],[]
        
        grid, grid_size = args.size//(args.gsize*2), args.gsize*2
       
        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]
        pred_conf = pred_collect[1][3][:,:,4,:,:].data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()
        conf_collect.append(max_conf_ii[0])
        (best_n, gj, gi) = np.where(pred_conf[0,:,:,:] == max_conf_ii[0])
        best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
        pred_gi.append(gi)
        pred_gj.append(gj)
        pred_best_n.append(best_n)
        pred_bbox[0,0] = F.sigmoid(pred_collect[1][3][0, best_n, 0, gj, gi]) + gi
        pred_bbox[0,1] = F.sigmoid(pred_collect[1][3][0, best_n, 1, gj, gi]) + gj
        pred_bbox[0,2] = torch.exp(pred_collect[1][3][0, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[0,3] = torch.exp(pred_collect[1][3][0, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
        pred_bbox[0,:] = pred_bbox[0,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)

        ## fine 
        pred_conf_fine = pred_collect[0][3][:,:,4,:,:].contiguous().view(1,-1)
        max_conf, max_loc = torch.max(pred_conf_fine, dim=1)

        pred_bbox_fine = torch.zeros(1,4)
        pred_gi, pred_gj, pred_best_n = [],[],[]
        grid, grid_size = args.size//args.gsize, args.gsize

        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]
        pred_conf_fine = pred_collect[0][3][:,:,4,:,:].data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()
        conf_collect.append(max_conf_ii[0])
        (best_n, gj, gi) = np.where(pred_conf_fine[0,:,:,:] == max_conf_ii[0])
        best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
        pred_gi.append(gi)
        pred_gj.append(gj)
        pred_best_n.append(best_n)
        pred_bbox_fine[0,0] = F.sigmoid(pred_collect[0][3][0, best_n, 0, gj, gi]) + gi
        pred_bbox_fine[0,1] = F.sigmoid(pred_collect[0][3][0, best_n, 1, gj, gi]) + gj
        pred_bbox_fine[0,2] = torch.exp(pred_collect[0][3][0, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox_fine[0,3] = torch.exp(pred_collect[0][3][0, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
        pred_bbox_fine[0,:] = pred_bbox_fine[0,:] * grid_size
        pred_bbox_fine = xywh2xyxy(pred_bbox_fine)


        target_bbox = bbox.data.cpu()
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio

        pred_bbox_fine[:,0], pred_bbox_fine[:,2] = (pred_bbox_fine[:,0]-dw)/ratio, (pred_bbox_fine[:,2]-dw)/ratio
        pred_bbox_fine[:,1], pred_bbox_fine[:,3] = (pred_bbox_fine[:,1]-dh)/ratio, (pred_bbox_fine[:,3]-dh)/ratio

        pred_bbox_8x8[:,0], pred_bbox_8x8[:,2] = (pred_bbox_8x8[:,0]-dw)/ratio, (pred_bbox_8x8[:,2]-dw)/ratio
        pred_bbox_8x8[:,1], pred_bbox_8x8[:,3] = (pred_bbox_8x8[:,1]-dh)/ratio, (pred_bbox_8x8[:,3]-dh)/ratio

        target_bbox[:,0], target_bbox[:,2] = (target_bbox[:,0]-dw)/ratio, (target_bbox[:,2]-dw)/ratio
        target_bbox[:,1], target_bbox[:,3] = (target_bbox[:,1]-dh)/ratio, (target_bbox[:,3]-dh)/ratio

        ## convert pred, gt box to original scale with meta-info
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)

        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        ## also revert image for visualization
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))

        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
        pred_bbox_fine[:,:2], pred_bbox_fine[:,2], pred_bbox_fine[:,3] = \
            torch.clamp(pred_bbox_fine[:,:2], min=0), torch.clamp(pred_bbox_fine[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox_fine[:,3], max=img_np.shape[2])
        pred_bbox_8x8[:,:2], pred_bbox_8x8[:,2], pred_bbox_8x8[:,3] = \
            torch.clamp(pred_bbox_8x8[:,:2], min=0), torch.clamp(pred_bbox_8x8[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox_8x8[:,3], max=img_np.shape[2])
        target_bbox[:,:2], target_bbox[:,2], target_bbox[:,3] = \
            torch.clamp(target_bbox[:,:2], min=0), torch.clamp(target_bbox[:,2], max=img_np.shape[3]), torch.clamp(target_bbox[:,3], max=img_np.shape[2])

        iou_8 = bbox_iou(pred_bbox_8x8, target_bbox.data.cpu(), x1y1x2y2=True)
        iou_16 = bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        iou_32 = bbox_iou(pred_bbox_fine, target_bbox.data.cpu(), x1y1x2y2=True)

        accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/1
        accu_8 = np.sum(np.array((iou_8.data.cpu().numpy()>0.5),dtype=float))/1
        accu_16 = np.sum(np.array((iou_16.data.cpu().numpy()>0.5),dtype=float))/1
        accu_32 = np.sum(np.array((iou_32.data.cpu().numpy()>0.5),dtype=float))/1
        acc_8.update(accu_8, imgs.size(0))
        acc_16.update(accu_16, imgs.size(0))
        acc_32.update(accu_32, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu {acc_8.val:.4f} ({acc_8.avg:.4f})\t' \
                'Accu {acc_16.val:.4f} ({acc_16.avg:.4f})\t' \
                'Accu {acc_32.val:.4f} ({acc_32.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, acc_8=acc_8, acc_16=acc_16, acc_32=acc_32)
            print(print_str)
            logging.info(print_str)

    acc_max = max([acc_8.avg, acc_16.avg, acc_32.avg])
    print("Accu: %f, accu_8x8: %f, accu_16x16: %f, accu_32x32: %f" % (acc_max, acc_8.avg, acc_16.avg, acc_32.avg))
    logging.info("Accu: %f, accu_8x8: %f, accu_16x16: %f, accu_32x32: %f" % (acc_max, acc_8.avg, acc_16.avg, acc_32.avg))
    return acc_max


if __name__ == "__main__":
    main()
