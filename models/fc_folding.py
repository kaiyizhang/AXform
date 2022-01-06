import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import os
import time
from datetime import datetime, timedelta
from visdom import Visdom

from utils.utils import *
from utils.data_loader import ShapeNetCorev2PC2048Dataset


class PointNet(nn.Module):
    def __init__(self, K1):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, K1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(K1)
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, dim=2)
        return x


class FC_based(nn.Module):
    def __init__(self):
        super(FC_based, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 2048*3)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(-1, 2048, 3)
        return x


class Folding_based(nn.Module):
    def __init__(self):
        super(Folding_based, self).__init__()
        self.grid_size = 46

        self.conv1 = nn.Conv1d(512+2, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(3)

        self.conv4 = nn.Conv1d(512+3, 512, 1)
        self.conv5 = nn.Conv1d(512, 512, 1)
        self.conv6 = nn.Conv1d(512, 3, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(3)

    def forward(self, x):
        grid = torch.meshgrid([torch.linspace(-0.5, 0.5, self.grid_size),
                               torch.linspace(-0.5, 0.5, self.grid_size)])
        grid_ = torch.cat((grid[1].reshape(self.grid_size**2, 1), grid[0].reshape(self.grid_size**2, 1)), 1).to(x.device)
        grid_ = torch.unsqueeze(grid_, 0)
        grid_feature = grid_.repeat(x.shape[0], 1, 1)  # B*2116*2

        feature = x.reshape(-1, 1, x.shape[1]).repeat(1, 2116, 1)

        x = torch.cat((feature, grid_feature), dim=2)
        
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x_1 = x.transpose(1, 2).contiguous()

        x = torch.cat((feature, x_1), dim=2)
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x_2 = x.transpose(1, 2).contiguous()
        return x_1, x_2


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.opt = opt

        if opt.method == 'fc-based':
            self.pointnet = PointNet(K1=1024)
            self.decoder = FC_based()
        if opt.method == 'folding-based':
            self.pointnet = PointNet(K1=512)
            self.decoder = Folding_based()

    def forward(self, x):
        if self.opt.method == 'fc-based':
            x_feat = self.pointnet(x)
            x = self.decoder(x_feat)
            return [x]

        if self.opt.method == 'folding-based':
            x_feat = self.pointnet(x)
            x1, x2 = self.decoder(x_feat)
            x1 = farthest_point_sample(x1, 2048)
            x2 = farthest_point_sample(x2, 2048)
            return [x1, x2]


class Runner(nn.Module):
    def __init__(self, opt):
        super(Runner, self).__init__()
        self.opt = opt
        self.network = torch.nn.DataParallel(Network(opt).to(opt.device), device_ids=opt.gpu_ids)
        self.display_id = opt.display_id
        self.vis = Visdom(env='%s' % self.display_id+'_'+str(opt.gpu_ids[0]))

        self.loss_cd = L2_ChamferLoss()
        self.eval_cd = L2_ChamferEval()
        self.loss_emd = EMDLoss()
        self.eval_emd = EMDEval()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=opt.lr, betas=(0.9, 0.999))
        
        self.best_cd = 0
        self.cd = {'02691156': {'num': 0, 'value': 0},
                   '02958343': {'num': 0, 'value': 0},
                   '03001627': {'num': 0, 'value': 0},
                   'overall' : {'num': 0, 'value': 0}}
        self.emd = {'02691156': {'num': 0, 'value': 0},
                    '02958343': {'num': 0, 'value': 0},
                    '03001627': {'num': 0, 'value': 0},
                    'overall' : {'num': 0, 'value': 0}}
        self.id2cat = {'02691156': 'airplane',
                       '02958343': 'car',
                       '03001627': 'chair',
                       'overall' : 'overall'}
        self.cat2id = {v: k for k, v in self.id2cat.items()}

    def train_model(self, data, args):
        start = time.time()

        foldername = data[0]
        filename = data[1]
        input = data[2].to(self.opt.device)
        
        self.network.train()
        output = self.network(input)
        self.loss = self.loss_cd(output[-1], input)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        end = time.time()
        if args[1] % 20 == 0:
            print('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]), end=' ')
            print('Loss: %.6f Time: %.6f' % (self.loss, end - start))
            with open(os.path.join(self.opt.save_path, 'runlog.txt'), 'a') as f:
                f.write('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]))
                f.write('Loss: %.6f Time: %.6f\n' % (self.loss, end - start))

            if self.opt.visual:
                plot_diff_pcds([input[0]]+[output[i][0] for i in range(len(output))],
                                vis=self.vis,
                                title='train for epoch %d' % args[0],
                                legend=['input']+['output'+str(i) for i in range(len(output))],
                                win='train_vis')

    def val_model(self, data, args):
        with torch.no_grad():
            foldername_val = data[0]
            filename_val = data[1]
            input_val = torch.unsqueeze(data[2], 0).to(self.opt.device)
            
            self.network.eval()
            output_val = self.network(input_val)

            value = self.eval_cd(output_val[-1], input_val)
            self.cd[foldername_val]['num'] += 1
            self.cd['overall']['num'] += 1
            self.cd[foldername_val]['value'] += value
            self.cd['overall']['value'] += value
        
        if self.opt.visual:
            if args[1] % 200 == 0:
                plot_diff_pcds([input_val[0]]+[output_val[i][0] for i in range(len(output_val))],
                                vis=self.vis,
                                title=foldername_val+'_'+filename_val,
                                legend=['input']+['output'+str(i) for i in range(len(output_val))],
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
            self.opt.class_choice = ['airplane', 'car', 'chair']
        self.opt.class_choice.append('overall')
        for i in range(len(self.opt.class_choice)):
            cat_name = self.opt.class_choice[i]
            losses['val_'+cat_name] = self.cd[self.cat2id[cat_name]]['value'].item()
        plot_loss_curves(self, losses, vis=self.vis, win='loss_curves')

        save_ckpt(self, step=20)
        self.cd = {'02691156': {'num': 0, 'value': 0},
                   '02958343': {'num': 0, 'value': 0},
                   '03001627': {'num': 0, 'value': 0},
                   'overall' : {'num': 0, 'value': 0}}
    
    def load_pretrained(self):
        ckpt = torch.load(self.opt.ckpt_path, map_location=lambda storage, location: storage)
        print("load epoch: %d" % (ckpt['epoch']))
        self.network.load_state_dict(ckpt['state_dict'])
    
    def calc_params(self):
        torchsummary.summary(self.network, input_size=(2048, 3))
        print('Param. (wo PointNet):', sum(p.numel() for p in self.network.module.decoder.parameters() if p.requires_grad))

    def calc_emd(self, data):
        with torch.no_grad():
            foldername_val = data[0]
            filename_val = data[1]
            input_val = torch.unsqueeze(data[2], 0).to(self.opt.device)
            
            self.network.eval()
            output_val = self.network(input_val)

            value = self.eval_emd(output_val[-1], input_val)
            self.emd[foldername_val]['num'] += 1
            self.emd['overall']['num'] += 1
            self.emd[foldername_val]['value'] += value
            self.emd['overall']['value'] += value

    def print_emd(self):
        print('emd result:')
        for key in self.emd:
            self.emd[key]['value'] /= max(self.emd[key]['num'], 1)
            print(self.id2cat[key]+': %.6f' % (self.emd[key]['value']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=list, default=[0], help='gpu_ids seperated by comma')
    parser.add_argument('--mode', type=str, default='train', help='train | test')
    parser.add_argument('--method', type=str, default='fc-based', help='fc-based | folding-based')
    parser.add_argument('--class_choice', default=['airplane'], help='category names | None')
    parser.add_argument('--visual', type=bool, default=True, help='visualization during training')
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoints')

    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=200, help='the epoch number of training the model')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--dataroot', default='./data/ShapeNetCore.v2.PC2048', help='path to point clouds')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=str, default='gpu', help='window id of the web display')

    opt = parser.parse_args()
    torch.cuda.set_device('cuda:'+str(opt.gpu_ids[0]))
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ShapeNetCorev2PC2048Dataset(root=opt.dataroot, class_choice=opt.class_choice, split='train')
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=True,
                                                num_workers=opt.nThreads)

    val_dataset = ShapeNetCorev2PC2048Dataset(root=opt.dataroot, class_choice=opt.class_choice, split='test')

    if opt.mode == "train":
        now = (datetime.utcnow()+timedelta(hours=8)).isoformat()
        opt.save_path = os.path.join('./log', 'fc_folding', now)
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        os.system('cp ./models/fc_folding.py %s' % opt.save_path)

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
            if epoch <= 50:
                opt.lr = 0.0001
            elif epoch <= 150:
                opt.lr = 0.00001
            else:
                opt.lr = 0.000001

            for i, data in enumerate(dataloader):
                runner.train_model(data=data, args=[epoch, i+1, len(dataloader)])
            for i in range(len(val_dataset)):
                runner.val_model(data=val_dataset[i], args=[epoch, i+1])
            runner.after_one_epoch(args=[epoch])

    if opt.mode == "test":
        runner = Runner(opt)
        print("calc params...")
        runner.calc_params()

        print("calc emd...")
        runner.load_pretrained()
        for i in range(len(val_dataset)):
            runner.calc_emd(data=val_dataset[i])
        runner.print_emd()