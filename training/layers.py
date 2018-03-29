import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
    
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class AttributeOutput(nn.Module):
    def __init__(self):
        super(AttributeOutput, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=2),
                              nn.BatchNorm3d(64),
                              nn.ReLU(),
                              nn.Conv3d(64, 64, kernel_size=1),
                              nn.ReLU()
                              )
        self.conv1 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=2),
                              nn.BatchNorm3d(64),
                              nn.ReLU(),
                              nn.Conv3d(64, 64, kernel_size=1),
                              nn.ReLU(),
                              nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
                              )
        self.conv2 = nn.Sequential(nn.Conv3d(128, 64, kernel_size=2),
                              nn.BatchNorm3d(64),
                              nn.ReLU(),
                              nn.Conv3d(64, 64, kernel_size=1),
                              nn.ReLU(),
                              nn.AvgPool3d((4, 4, 4), stride=(4, 4, 4))
                              )
        self.fc = nn.Sequential(nn.Linear(1536, 256),
                           nn.Linear(256, 128),
                           nn.Linear(128, 1))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        x0_drop = self.drop(x[0])
        x1_drop = self.drop(x[1])
        x2_drop = self.drop(x[2])
        conv0 = self.conv0(x0_drop)
        conv1 = self.conv1(x1_drop)
        conv2 = self.conv2(x2_drop)
        conv0 = conv0.view(conv0.size(0), -1)
        conv1 = conv1.view(conv1.size(0), -1)
        conv2 = conv2.view(conv2.size(0), -1)
        out = torch.cat((conv0, conv1, conv2), 1)
        out = self.fc(out)

        return out

class Rec3(nn.Module):
    def __init__(self, n0, n1, n2, n3, p = 0.0, integrate = True):
        super(Rec3, self).__init__()
        
        self.block01 = nn.Sequential(
            nn.Conv3d(n0, n1, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))

        self.block11 = nn.Sequential(
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
        
        self.block21 = nn.Sequential(
            nn.ConvTranspose3d(n2, n1, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
 
        self.block12 = nn.Sequential(
            nn.Conv3d(n1, n2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block22 = nn.Sequential(
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block32 = nn.Sequential(
            nn.ConvTranspose3d(n3, n2, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
 
        self.block23 = nn.Sequential(
            nn.Conv3d(n2, n3, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.block33 = nn.Sequential(
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.relu = nn.ReLU(inplace = True)
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            coef = torch.bernoulli((1.0 - self.p) * torch.ones(8))
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)

def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

class MAE_Loss(nn.Module):
    def __init__(self):
        super(MAE_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, output, labels, train=True):
        if not (train):
            #calcification max range 5, internal structure max range 4, other max range 5
            for i in range(9):
                if (i == 5):
                    if (output[:, i].data[0] > 6):
                        output[:, i].data[0] = 6.0
                elif (i == 6):
                    if (output[:, i].data[0] > 4):
                        output[:, i].data[0] = 4.0
                else:
                    if (output[:, i].data[0] > 5):
                        output[:, i].data[0] = 5.0

                if (output[:, i].data[0] < 1):
                    output[:, i].data[0] = 1.0

        criteria = self.l1_loss
        losses = [
            criteria(output[:,0], labels[:,0]),
            criteria(output[:,1], labels[:,1]),
            criteria(output[:,2], labels[:,2]),
            criteria(output[:,3], labels[:,3]),
            criteria(output[:,4], labels[:,4]),
            criteria(output[:,5], labels[:,5]),
            criteria(output[:,6], labels[:,6]),
            criteria(output[:,7], labels[:,7]),
            criteria(output[:,8], labels[:,8]),
        ]

        total_loss = 0
        for loss in losses:
            total_loss += loss
        return total_loss, losses

class MAE_Loss_one(nn.Module):
    def __init__(self):
        super(MAE_Loss_one, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, output, labels, train=True):
        if not (train):
            #calcification max range 5, internal structure max range 4, other max range 5
                if (output[:, 0].data[0] > 5):
                    output[:, 0].data[0] = 5.0

                if (output[:, 0].data[0] < 1):
                    output[:, 0].data[0] = 1.0

        criteria = self.l1_loss

        total_loss = criteria(output[:,0], labels[:,0])

        return total_loss

class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output,thresh = -3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx,yy,zz,aa = np.where(mask)

        output = output[xx,yy,zz,aa]
        #bboxes = nms(output, 0.4)
        if ismask:
            return output, [xx,yy,zz,aa]
        else:
            return output

        #output = output[output[:, 0] >= self.conf_th] 
        #bboxes = nms(output, self.nms_th)
def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def acc(pbb, lbb, conf_th, nms_th, detect_th):
    if (len(pbb) > 0):
        pbb = pbb[pbb[:, 0] >= conf_th]
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score>bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p,[bestscore]],0))
            else:
                fp.append(np.concatenate([p,[bestscore]],0))
        if flag == 0:
            fp.append(np.concatenate([p,[bestscore]],0))
    for i,l in enumerate(lbb):
        if l_flag[i]==0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5],l))
            if len(score)!=0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore<detect_th:
                fn.append(np.concatenate([l,[bestscore]],0))

    return tp, fp, fn, len(lbb)

def topkpbb(pbb,lbb,nms_th,detect_th,topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp)+len(fp)<topk:
        conf_th = conf_th-0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th<-3:
            break
    tp = np.array(tp).reshape([len(tp),6])
    fp = np.array(fp).reshape([len(fp),6])
    fn = np.array(fn).reshape([len(fn),5])
    allp  = np.concatenate([tp,fp],0)
    sorting = np.argsort(allp[:,0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk,len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
#     print(fp_in_topk)
    fn_i =       np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp = allp[:topk]
    if len(fn_i)>0:
        fn = np.concatenate([fn,tp[fn_i,:5]])
    else:
        fn = fn
    if len(tp_in_topk)>0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk)>0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp , fn
