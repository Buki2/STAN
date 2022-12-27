from collections import OrderedDict
import math
import random
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models
from torch.nn.init import kaiming_normal, kaiming_uniform
from .darknet import ConvBatchNormReLU, ConvBatchNormReLU_3d


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init_params(m.weight)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        return (gammas * x) + betas


def mask_softmax(attn_score, word_mask, tempuature=10., clssep=False, lstm=False):
    word_mask_cp = word_mask[:,:attn_score.shape[1]].clone()
    score = F.softmax(attn_score*tempuature, dim=1)
    if not clssep:
        for ii in range(word_mask_cp.shape[0]):
            if lstm:
                word_mask_cp[ii,word_mask_cp[ii,:].sum().long()-1]=0
            else:
                word_mask_cp[ii,0]=0
                word_mask_cp[ii,word_mask_cp[ii,:].sum().long()]=0 ## set one to 0 already
    mask_score = score * word_mask_cp.float()
    mask_score = mask_score/(mask_score.sum(1)+1e-8).view(mask_score.size(0), 1).expand(mask_score.size(0), mask_score.size(1))
    return mask_score


class cross_att_blocked(nn.Module):
    def __init__(self,emb_size=512,raw_feature_norm='softmax',head=3):
        super(cross_att_blocked, self).__init__()

        self.head = head
        self.raw_feature_norm = raw_feature_norm
        #visual
        modules = OrderedDict()
        self.attributes = nn.ModuleDict()
        for n in range(head):
            modules['attribute_%d'%n] = nn.Linear(emb_size+8, emb_size)
        self.attributes.update(modules)

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, fvisu, raw_fword, coord, att_weights_txt, att_weights_visu,word_mask):
        batch_size = fvisu.size(0)
        map_fvisu_orig = fvisu
        map_fvisu = torch.cat([map_fvisu_orig,coord],dim=-1)

        num_attributes_visu = []
        for n in range(self.head):
            map_att = self.relu(self.dropout(self.attributes['attribute_%d'%n](map_fvisu)))
            num_attributes_visu.append(map_att)

        att_attributes_visu = []
        cosine_weights_visu = []
        cosine_weights_txt = []
        att_weights_visu_collections = []
        
        raw_fword = l2norm(raw_fword,dim=2)
        
        for i,f in enumerate(num_attributes_visu):
            f = l2norm(f,dim=2)
            att_f_visu, att_weights_visu, cosine_att_visu = func_attention(f, raw_fword, self.raw_feature_norm,9,weight=att_weights_visu)
            att_weights_visu_collections.append(att_weights_visu)
            att_weights_visu = torch.stack(att_weights_visu_collections,-1).max(-1)[0]
            att_attributes_visu.append(att_f_visu)
            
            cosine_weights_visu.append(cosine_att_visu)
            
        att_attributes_visu = torch.stack(att_attributes_visu,dim=-1).sum(-1)

        att_weights_visu_collections = torch.stack(att_weights_visu_collections,dim=-1).max(-1)[0]
        num_attributes_txt = [raw_fword]
        att_attributes_txt = []
        att_weights_txt_collections = []
        map_fvisu_orig = l2norm(map_fvisu_orig,dim=2)
        for i,f in enumerate(num_attributes_txt):
            f = l2norm(f,dim=2)
            att_f_txt, att_weights_txt, cosine_att_txt = func_attention(f, map_fvisu_orig,self.raw_feature_norm,9,weight=att_weights_txt)
            att_weights_txt_collections.append(att_weights_txt)
            att_weights_txt = torch.stack(att_weights_txt_collections,-1).max(-1)[0]
            att_attributes_txt.append(att_f_txt)
            cosine_weights_txt.append(cosine_att_txt)
        att_attributes_txt = torch.stack(att_attributes_txt,dim=-1).sum(-1)

        att_weights_txt_collections = torch.stack(att_weights_txt_collections,dim=-1).max(-1)[0]

        att_attributes_visu = l2norm(att_attributes_visu,dim=2)
        att_attributes_txt = l2norm(att_attributes_txt,dim=2)
        return att_attributes_visu, att_attributes_txt, cosine_weights_txt, cosine_weights_visu, att_weights_txt_collections, att_weights_visu_collections


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_head=10, emb_size=512):
        super().__init__()
        self.num_pos_head = num_pos_head
        self.pos_embed = nn.Embedding(num_pos_head, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        i = torch.arange(self.num_pos_head, device=x.device)
        pos = self.pos_embed(i)
        return pos.unsqueeze(0)


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(0)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X+eps, 2).sum(dim=dim, keepdim=True).sqrt() + eps
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
    
    if weight is not None and weight.size(1) == attn.size(1) and weight.size(2) == attn.size(2):
      attn = attn*(1-weight)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    cosines_attn = attn.max(2)[0]
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    attn_out = attn.clone()
    attn_out = attn_out.view(batch_size, sourceL, queryL)

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

    return weightedContext, attn_out, cosines_attn
