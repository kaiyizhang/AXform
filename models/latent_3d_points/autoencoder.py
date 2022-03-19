import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from datetime import datetime, timedelta
from visdom import Visdom

from utils.utils import *
from utils.data_loader import ShapeNetCorev2PC2048Dataset


class Encoder(nn.Module):
    def __init__(self, bneck_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, bneck_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(bneck_size)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x, _ = torch.max(x, dim=2)
        return x


class Decoder(nn.Module):
    def __init__(self, bneck_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(bneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2048*3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 2048, 3)
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
        self.conv3 = nn.Conv1d(128, 2048, 1)
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
        self.encoder = Encoder(bneck_size=128)

        if opt.method == 'original':
            self.decoder = Decoder(bneck_size=128)
        if opt.method == 'axform':
            self.decoder = AXform(K1=128, K2=32, N=128)

    def forward(self, x):
        x_latent = self.encoder(x)
        x = self.decoder(x_latent)
        return [x]


class Runner:
    def __init__(self, opt):
        super(Runner, self).__init__()
        self.opt = opt
        self.network = torch.nn.DataParallel(Network(opt).to(opt.device), device_ids=opt.gpu_ids)
        self.display_id = opt.display_id
        self.vis = Visdom(env='%s' % self.display_id+'_'+str(opt.gpu_ids[0]))

        self.loss_cd = L2_ChamferLoss()
        self.eval_cd = L2_ChamferEval()
        # self.loss_emd = EMDLoss()
        # self.eval_emd = EMDEval()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=opt.lr, betas=(0.9, 0.999))
        
        self.best_cd = 0
        self.cd = {'02691156': {'num': 0, 'value': 0},
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
        if args[1] % 10 == 0:
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

        save_ckpt(self, step=50)
        self.cd = {'02691156': {'num': 0, 'value': 0},
                   '02958343': {'num': 0, 'value': 0},
                   '03001627': {'num': 0, 'value': 0},
                   'overall' : {'num': 0, 'value': 0}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=list, default=[0], help='gpu_ids seperated by comma')
    parser.add_argument('--mode', type=str, default='train', help='train...')
    parser.add_argument('--method', type=str, default='original', help='original | axform')
    parser.add_argument('--class_choice', default=['airplane'], help='category names | None')
    parser.add_argument('--visual', type=bool, default=True, help='visualization during training')

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=500, help='the epoch number of training the model')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
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
        opt.save_path = os.path.join('./log', 'latent_3d_points/autoencoder', now)
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        os.system('cp ./models/latent_3d_points/autoencoder.py %s' % opt.save_path)

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
            for i, data in enumerate(dataloader):
                runner.train_model(data=data, args=[epoch, i+1, len(dataloader)])
            for i in range(len(val_dataset)):
                runner.val_model(data=val_dataset[i], args=[epoch, i+1])
            runner.after_one_epoch(args=[epoch])