import argparse
import os
import random
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from waymo_dataset import WaymoSiameseDataset
from pointnet2.models import get_model
import kitty_utils as utils


class FocalLoss(nn.Module):
    def __init__(self, pos=1, alpha=0.3, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.pos = pos
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-7

    def forward(self, inputs, targets, masks=None): #(2,128) (2,128)
        if masks == None:
            inputs = inputs.view(-1) #(256)
            targets = targets.view(-1) #(256)
            pred = torch.sigmoid(inputs) #(256)

            pos = targets.data.eq(1).nonzero().squeeze().cuda()  #---------index
            neg = targets.data.eq(0).nonzero().squeeze().cuda()

            pred_pos = torch.index_select(pred, 0, pos)
            targets_pos = torch.index_select(targets, 0, pos)

            pred_neg = torch.index_select(pred, 0, neg)
            targets_neg = torch.index_select(targets, 0, neg)

            loss_pos = -1 * torch.pow((1-pred_pos),self.gamma) * torch.log(pred_pos + self.eps) * self.pos

            loss_neg = -1 * torch.pow((pred_neg),self.gamma) * torch.log(1 - pred_neg + self.eps)

            loss = torch.cat((loss_pos, loss_neg), dim=0)

            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss

            return loss
        else:
            inputs = inputs.view(-1)  # (256)
            targets = targets.view(-1)  # (256)
            masks = masks.view(-1)
            pred = torch.sigmoid(inputs)  # (256)

            pos = targets.data.eq(1).nonzero().squeeze().cuda()  # ---------index
            neg = targets.data.eq(0).nonzero().squeeze().cuda()
            mask_i = targets.data.gt(0).nonzero().squeeze().cuda()

            pred_pos = torch.index_select(pred, 0, pos)
            targets_pos = torch.index_select(targets, 0, pos)

            pred_neg = torch.index_select(pred, 0, neg)
            targets_neg = torch.index_select(targets, 0, neg)

            masks = torch.index_select(masks, 0, mask_i)


            loss_pos = -1 * torch.pow((1 - pred_pos), self.gamma) * torch.log(pred_pos + self.eps) * self.pos * (1 + masks)

            loss_neg = -1 * torch.pow((pred_neg), self.gamma) * torch.log(1 - pred_neg + self.eps)

            loss = torch.cat((loss_pos, loss_neg), dim=0)

            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss

            return loss


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--input_feature_num', type=int, default = 0,  help='number of input point features')
parser.add_argument('--data_dir', type=str, default = '/media/xxx/waymo_tracking',  help='dataset path')
parser.add_argument('--category_name', type=str, default = 'Vehicle',  help='Object to Track (Vehicle/Pedestrian/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
parser.add_argument('--tiny', type=bool, default=False)
parser.add_argument('--input_size', type=int, default=1024)
parser.add_argument('--save_interval', type=int, default=1)

opt = parser.parse_args()
print(opt)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = opt.save_root_dir
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)

logging.info('======================================================')


# 1. Load data
train_data = WaymoSiameseDataset(
    input_size=opt.input_size,
    path=os.path.join(opt.data_dir, 'train'),
    category_name=opt.category_name,
    min_seq_len=2,
    min_pts_in_gt=10,
    offset_BB=0.1,
    scale_BB=1.0) # opt.scale)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    pin_memory=True)

test_data = WaymoSiameseDataset(
    input_size=opt.input_size,
    path=os.path.join(opt.data_dir, 'test'),
    category_name=opt.category_name,
    min_seq_len=10,
    min_pts_in_gt=10,
    offset_BB=0.1,
    scale_BB=1.0)

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=int(opt.batchSize // 2),
    shuffle=False,
    num_workers=int(opt.workers),
    pin_memory=True)

print('#Train data:', len(train_data), '#Test data:', len(test_data))

# 2. Define model, loss and optimizer
netR = get_model(name='T', # opt.type,
                 input_channels=opt.input_feature_num,
                 use_xyz=True,
                 input_size=opt.input_size)
netR = torch.nn.DataParallel(netR)
if opt.model != '':
    netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))

netR.cuda()

criterion_cla_focal = FocalLoss(pos=1, gamma=2, size_average=True).cuda()
criterion_cla = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0])).cuda()
criterion_reg = nn.MSELoss(reduction='none').cuda()
criterion_objective = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0]),
    reduction='none').cuda()
criterion_objective_focal = FocalLoss(pos=2, gamma=2, size_average=True).cuda()
criterion_box = nn.MSELoss(reduction='none').cuda()

optimizer = optim.Adam(netR.parameters(),
                       lr=opt.learning_rate,
                       betas = (0.5, 0.999),
                       eps=1e-6)
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.2)

def one_sample_step(input_dict, model, optimizer, train=True):
    optimizer.zero_grad()
    output_dict = model(input_dict)

    label_cla = output_dict['cls_label']
    label_reg = output_dict['reg_label']

    classification_scores = output_dict['classification_scores']
    offsets = output_dict['offsets']  # vote xyz
    angles_scores = output_dict['angles_scores']
    center_xyz = output_dict['center_xyz']  # candi

    # loss_cla = criterion_cla(estimation_cla, label_cla)
    loss_cla_focal = criterion_cla_focal(classification_scores, label_cla)
    # loss_cla_focal = criterion_cla_ce(classification_scores, label_cla)

    # vote -----> box center
    loss_reg = criterion_reg(offsets, label_reg[:, :, 0:3])  # 16x128x3
    loss_reg = (loss_reg.mean(2) * label_cla * (1 + classification_scores.sigmoid())).sum() / (label_cla.sum() + 1e-06)

    centerness_label = utils.getlabelPC_train(PC=center_xyz, maxi=input_dict['maxi'], mini=input_dict['mini']).cuda()
    centerness_mask = utils.getmaskPC_train(PC=center_xyz, maxi=input_dict['maxi_xyz'],
                                            mini=input_dict['mini_xyz']).cuda()

    K = center_xyz.shape[1]  # 128
    B = center_xyz.shape[0]
    dist = torch.sum((center_xyz - label_reg[:, :, 0:3]) ** 2, dim=-1)
    index = torch.argmin(dist, dim=1, keepdim=True)
    for i in range(B):
        centerness_label[i, index[i]] = 1

    # objectness_mask = torch.ones((B, K), requires_grad=False).float().cuda()
    # loss_objective = criterion_objective(angles_scores[:, :, 1], centerness_label)
    # loss_objective = torch.sum(loss_objective * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
    loss_objective_focal = criterion_objective_focal(angles_scores[:, :, 1], centerness_label, centerness_mask)

    box_mask = label_cla
    loss_box = criterion_box(angles_scores[:, :, 0:1], label_reg[:, 0:K, 3:4])
    loss_box = (loss_box.mean(2) * box_mask).sum() / (box_mask.sum() + 1e-06)

    loss = loss_cla_focal + loss_reg + 1.0 * loss_box + 1.0 * loss_objective_focal

    if train:
        loss.backward()
        optimizer.step()

    classification_scores_cpu = classification_scores.sigmoid().detach().cpu().numpy()
    label_cla_cpu = label_cla.detach().cpu().numpy()
    correct = float(np.sum((
                                   classification_scores_cpu > 0.4) == label_cla_cpu)
                    ) / label_cla_cpu.size
    true_correct = float(np.sum(
        (np.float32(classification_scores_cpu > 0.4)
         + label_cla_cpu) == 2)) \
                   / np.sum(label_cla_cpu)

    return {
            'correct' : correct,
            'true_correct' : true_correct,
            'loss_cla' : loss_cla_focal,
            'loss_reg' : loss_reg,
            'loss' : loss
        }


# 3. Training and testing
for epoch in range(opt.nepoch):
    scheduler.step(epoch)
    print('======>>>>> Online epoch: #%d, lr=%f <<<<<======' %(epoch, scheduler.get_lr()[0]))
    # 3.1 switch to train mode
    # torch.cuda.synchronize()
    netR.train()
    train_mse = 0.0
    timer = time.time()

    batch_correct = 0.0
    batch_cla_loss = 0.0
    batch_reg_loss = 0.0
    batch_box_loss = 0.0
    batch_num = 0.0
    batch_iou = 0.0
    batch_true_correct = 0.0
    for i, input_dict in enumerate(tqdm(train_dataloader, 0)):
        if len(input_dict['search']) == 1:
            continue
        # torch.cuda.synchronize()
        # 3.1.1 load inputs and targets
        for k, v in input_dict.items():
            input_dict[k] = Variable(v, requires_grad=False).cuda()
        output_dict = one_sample_step(input_dict, netR, optimizer)

        correct = output_dict['correct']
        true_correct = output_dict['true_correct']
        loss_cla = output_dict['loss_cla']
        loss_reg = output_dict['loss_reg']
        loss = output_dict['loss']

        train_mse = train_mse + loss.data * len(input_dict['search'])
        batch_correct += correct
        batch_cla_loss += loss_cla.data
        batch_reg_loss += loss_reg.data
        batch_num += 1 # len(input_dict['search'])
        batch_true_correct += true_correct
        if (i + 1) % 20 == 0:
            print('\n ---- batch: %03d ----' % (i+1))
            print('cla_loss: %f, reg_loss: %f, box_loss: %f' %
                   (batch_cla_loss/20,batch_reg_loss/20,batch_box_loss/20))
            print('accuracy: %f' % (batch_correct / float(batch_num)))
            print('true accuracy: %f' % (batch_true_correct / float(batch_num)))
            batch_correct = 0.0
            batch_cla_loss = 0.0
            batch_reg_loss = 0.0
            batch_box_loss = 0.0
            batch_num = 0.0
            batch_true_correct = 0.0

    # time taken
    train_mse = train_mse / len(train_data)
    # torch.cuda.synchronize()
    timer = time.time() - timer
    timer = timer / len(train_data)
    print('==> time to learn 1 sample = %f (ms)' %(timer*1000))


    if epoch and (epoch % opt.save_interval == 0 or epoch == opt.nepoch-1):
        torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))

    # 3.2 switch to evaluate mode
    netR.eval()
    test_cla_loss = 0.0
    test_reg_loss = 0.0
    test_box_loss = 0.0
    test_correct = 0.0
    test_true_correct = 0.0
    timer = time.time()
    for i, data in enumerate(tqdm(test_dataloader, 0)):
        for k, v in input_dict.items():
            input_dict[k] = Variable(v, requires_grad=False).cuda()

        with torch.no_grad():
            output_dict = one_sample_step(input_dict, netR, optimizer, train=False)

        correct = output_dict['correct']
        true_correct = output_dict['true_correct']

        test_correct += correct
        test_true_correct += true_correct

    # time taken
    timer = time.time() - timer
    timer = timer / len(test_data)
    print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
    # print mse
    test_cla_loss = test_cla_loss / len(test_data)
    test_reg_loss = test_reg_loss / len(test_data)
    test_box_loss = test_box_loss / len(test_data)
    print('cla_loss: %f, reg_loss: %f, box_loss: %f, #test_data = %d' %
        (test_cla_loss, test_reg_loss, test_box_loss, len(test_data)))
    test_correct = test_correct / len(test_dataloader)
    print('mean-correct of 1 sample: %f, #test_data = %d' %(test_correct, len(test_data)))
    test_true_correct = test_true_correct / len(test_dataloader)
    print('true correct of 1 sample: %f' %(test_true_correct))
    # log
    logging.info('Epoch#%d: train error=%e, test error=%e,%e,%e, test correct=%e, %e, lr = %f' %
                (epoch, train_mse, test_cla_loss, test_reg_loss, test_box_loss,
                 test_correct, test_true_correct, scheduler.get_lr()[0]))

