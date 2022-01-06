import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
from datetime import datetime, timedelta
from visdom import Visdom

from utils.utils import *
from utils.data_loader import PCNDataset


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, dim=2)
        return x


class MappingNet(nn.Module):
    def __init__(self, K1):
        super(MappingNet, self).__init__()
        self.K1 = K1

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, self.K1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(self.K1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        return x


class AXform(nn.Module):
    def __init__(self, K1, K2, N):
        super(AXform, self).__init__()
        self.K1 = K1
        self.K2 = K2
        self.N = N  # N>=K2

        self.fc1 = nn.Linear(K1, N*K2)

        self.conv1 = nn.Conv1d(K2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=2)

        self.conv4 = nn.Conv1d(K2, 3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.N, self.K2)

        x_base = x
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x_weights = self.softmax(x)
        x = torch.bmm(x_weights, x_base)

        x = x.transpose(1, 2).contiguous()
        x = self.conv4(x)
        x = x.transpose(1, 2).contiguous()
        return x


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.opt = opt
        self.num_branch = 16
        self.K1 = 128
        self.K2 = 32
        self.N = 128

        self.pointnet = PointNet()
        self.featmap = nn.ModuleList([MappingNet(self.K1) for i in range(self.num_branch)])
        self.pointgen = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])
        if opt.method == 'integrated':
            self.pointref = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])

    def forward(self, x):
        if self.opt.method == 'vanilla':
            x_feat = self.pointnet(x)

            x_1 = torch.empty(size=(x.shape[0], 0, 3)).to(x.device)
            for i in range(self.num_branch):
                _x_1 = self.pointgen[i](self.featmap[i](x_feat))
                x_1 = torch.cat((x_1, _x_1), dim=1)
            return [x_1]

        if self.opt.method == 'integrated':
            x_partial = x
            x_partial = farthest_point_sample(x_partial, 512)

            x_feat = self.pointnet(x)

            x_1 = torch.empty(size=(x.shape[0], 0, 3)).to(x.device)
            x_2 = torch.empty(size=(x.shape[0], 0, 3)).to(x.device)
            for i in range(self.num_branch):
                _x_1 = self.pointgen[i](self.featmap[i](x_feat))
                x_1 = torch.cat((x_1, _x_1), dim=1)

                x_base = torch.cat((farthest_point_sample(_x_1, 512), x_partial), dim=1)
                _x_2 = self.pointref[i](self.featmap[i](x_feat))
                _x_2 = x_base + _x_2
                x_2 = torch.cat((x_2, _x_2), dim=1)
            return [x_1, x_2]


class Runner(nn.Module):
    def __init__(self, opt):
        super(Runner, self).__init__()
        self.opt = opt
        self.network = torch.nn.DataParallel(Network(opt).to(self.opt.device), device_ids=self.opt.gpu_ids)
        self.display_id = self.opt.display_id
        self.vis = Visdom(env='%s' % self.display_id+'_'+str(self.opt.gpu_ids[0]))

        self.loss_cd = L1_ChamferLoss()
        self.eval_cd = L1_ChamferEval()
        self.eval_f1 = F1Score()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.opt.lr, betas=(0.9, 0.999))

        self.best_cd = 0
        self.cd = {'02691156': {'num': 0, 'value': 0},
                   '02933112': {'num': 0, 'value': 0},
                   '02958343': {'num': 0, 'value': 0},
                   '03001627': {'num': 0, 'value': 0},
                   '03636649': {'num': 0, 'value': 0},
                   '04256520': {'num': 0, 'value': 0},
                   '04379243': {'num': 0, 'value': 0},
                   '04530566': {'num': 0, 'value': 0},
                   'overall' : {'num': 0, 'value': 0}}
        self.f1 = {'02691156': {'num': 0, 'value': 0},
                   '02933112': {'num': 0, 'value': 0},
                   '02958343': {'num': 0, 'value': 0},
                   '03001627': {'num': 0, 'value': 0},
                   '03636649': {'num': 0, 'value': 0},
                   '04256520': {'num': 0, 'value': 0},
                   '04379243': {'num': 0, 'value': 0},
                   '04530566': {'num': 0, 'value': 0},
                   'overall' : {'num': 0, 'value': 0}}
        self.id2cat = {'02691156': 'airplane',
                       '02933112': 'cabinet',
                       '02958343': 'car',
                       '03001627': 'chair',
                       '03636649': 'lamp',
                       '04256520': 'sofa',
                       '04379243': 'table',
                       '04530566': 'vessel',
                       'overall' : 'overall'}
        self.cat2id = {v: k for k, v in self.id2cat.items()}

    def train_model(self, data, args):
        start = time.time()

        foldername = data[0]
        filename = data[1]
        partial = data[2].to(self.opt.device)
        gt = data[3].to(self.opt.device)  # B*16384*3
        
        self.network.train()
        complete = self.network(partial)
        if self.opt.method == 'vanilla':
            self.loss = self.loss_cd(complete[-1], gt)
        if self.opt.method == 'integrated':
            self.loss = self.loss_cd(complete[-1], gt) + self.opt.alpha * self.loss_cd(complete[-2], gt)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        end = time.time()
        if args[1] % 100 == 0:
            print('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]), end=' ')
            print('Loss: %.6f Time: %.6f' % (self.loss, end - start))
            with open(os.path.join(self.opt.save_path, 'runlog.txt'), 'a') as f:
                f.write('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]))
                f.write('Loss: %.6f Time: %.6f\n' % (self.loss, end - start))

            if self.opt.visual:
                plot_diff_pcds([partial[0], gt[0]]+[complete[i][0] for i in range(len(complete))],
                                vis=self.vis,
                                title='train for epoch %d' % args[0],
                                legend=['partial', 'gt']+['complete'+str(i) for i in range(len(complete))],
                                win='train_vis')

    def val_model(self, data, args):
        with torch.no_grad():
            foldername_val = data[0]
            filename_val = data[1]
            partial_val = torch.unsqueeze(data[2], 0).to(self.opt.device)
            gt_val = torch.unsqueeze(data[3], 0).to(self.opt.device)
            
            self.network.eval()
            complete_val = self.network(partial_val)

            value = self.eval_cd(complete_val[-1], gt_val)
            self.cd[foldername_val]['num'] += 1
            self.cd['overall']['num'] += 1
            self.cd[foldername_val]['value'] += value
            self.cd['overall']['value'] += value

        if self.opt.visual:
            if args[1] % 100 == 0:
                plot_diff_pcds([partial_val[0], gt_val[0]]+[complete_val[i][0] for i in range(len(complete_val))],
                                vis=self.vis,
                                title=foldername_val+'_'+filename_val,
                                legend=['partial', 'gt']+['complete'+str(i) for i in range(len(complete_val))],
                                win='val_vis'+foldername_val)

    def after_one_epoch(self, args):
        self.epoch = args[0]
        
        print('val result:')
        with open(os.path.join(self.opt.save_path, 'runlog.txt'), 'a') as f:
            f.write('val result:\n')
        for key in self.cd:
            self.cd[key]['value'] /= max(self.cd[key]['num'], 1)
            print(self.id2cat[key]+': CD: %.6f' % (self.cd[key]['value']))
            with open(os.path.join(self.opt.save_path, 'runlog.txt'), 'a') as f:
                f.write(self.id2cat[key]+': CD: %.6f\n' % (self.cd[key]['value']))

        losses = {'loss': self.loss.item()}
        if self.opt.class_choice is None:
            self.opt.class_choice = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
        self.opt.class_choice.append('overall')
        for i in range(len(self.opt.class_choice)):
            cat_name = self.opt.class_choice[i]
            losses['val_'+cat_name] = self.cd[self.cat2id[cat_name]]['value'].item()
        plot_loss_curves(self, losses, vis=self.vis, win='loss_curves')

        save_ckpt(self, step=10)
        self.cd = {'02691156': {'num': 0, 'value': 0},
                   '02933112': {'num': 0, 'value': 0},
                   '02958343': {'num': 0, 'value': 0},
                   '03001627': {'num': 0, 'value': 0},
                   '03636649': {'num': 0, 'value': 0},
                   '04256520': {'num': 0, 'value': 0},
                   '04379243': {'num': 0, 'value': 0},
                   '04530566': {'num': 0, 'value': 0},
                   'overall' : {'num': 0, 'value': 0}}
    
    def load_pretrained(self):
        ckpt = torch.load(self.opt.ckpt_path, map_location=lambda storage, location: storage)
        print("load epoch: %d" % (ckpt['epoch']))
        self.network.load_state_dict(ckpt['state_dict'])

    def calc_f1(self, data):
        with torch.no_grad():
            foldername_val = data[0]
            filename_val = data[1]
            partial_val = torch.unsqueeze(data[2], 0).to(self.opt.device)
            gt_val = torch.unsqueeze(data[3], 0).to(self.opt.device)
            
            self.network.eval()
            complete_val = self.network(partial_val)

            value, _, _ = self.eval_f1(complete_val[-1], gt_val)
            self.f1[foldername_val]['num'] += 1
            self.f1['overall']['num'] += 1
            self.f1[foldername_val]['value'] += value
            self.f1['overall']['value'] += value

    def print_f1(self):
        print('f1 result:')
        for key in self.f1:
            self.f1[key]['value'] /= max(self.f1[key]['num'], 1)
            print(self.id2cat[key]+': %.6f' % (self.f1[key]['value']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=list, default=[0], help='gpu_ids seperated by comma')
    parser.add_argument('--mode', type=str, default='train', help='train | test')
    parser.add_argument('--method', type=str, default='integrated', help='vanilla | integrated')
    parser.add_argument('--class_choice', default=None, help='category names | None')
    parser.add_argument('--visual', type=bool, default=True, help='visualization during training')
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoints')

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='the epoch number of training the model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--dataroot', default='./data/PCN/ShapeNet', help='path to point clouds')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=str, default='gpu', help='window id of the web display')

    opt = parser.parse_args()
    torch.cuda.set_device('cuda:'+str(opt.gpu_ids[0]))
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PCNDataset(root=opt.dataroot, class_choice=opt.class_choice, split='train')
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                num_workers=opt.nThreads)

    val_dataset = PCNDataset(root=opt.dataroot, class_choice=opt.class_choice, split='test')

    if opt.mode == "train":
        now = (datetime.utcnow()+timedelta(hours=8)).isoformat()
        opt.save_path = os.path.join('./log', 'axformnet', now)
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        os.system('cp axformnet.py %s' % opt.save_path)

        set_seed(42)
        
        print('------------ Options -------------')
        for k, v in sorted(vars(opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        with open(os.path.join(opt.save_path, 'runlog.txt'), 'a') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(vars(opt).items()):
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')

        runner = Runner(opt)
        for epoch in range(1, opt.n_epochs+1):
            if epoch < 25:
                opt.lr = 0.001
            elif epoch < 75:
                opt.lr = 0.0001
            else:
                opt.lr = 0.00001

            if epoch < 5:
                opt.alpha = 0.01
            elif epoch < 10:
                opt.alpha = 0.1
            elif epoch < 25:
                opt.alpha = 0.5
            else:
                opt.alpha = 1.0

            for i, data in enumerate(dataloader):
                runner.train_model(data=data, args=[epoch, i+1, len(dataloader)])
            for i in range(len(val_dataset)):
                runner.val_model(data=val_dataset[i], args=[epoch, i+1])
            runner.after_one_epoch(args=[epoch])

    if opt.mode == "test":
        runner = Runner(opt)
        runner.load_pretrained()
        for i in range(len(val_dataset)):
            runner.calc_f1(data=val_dataset[i])
        runner.print_f1()