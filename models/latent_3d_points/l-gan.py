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
from pprint import pprint, pformat

from utils.utils import *
from utils.data_loader import ShapeNetCorev2PC2048Dataset
from utils.metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from utils.metrics.evaluation_metrics import compute_all_metrics

from models.latent_3d_points.autoencoder import Network as Autoencoder


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):  # B*128
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # B*128


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):  # B*128
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        d_logit = self.fc3(x)
        return d_logit  # B*1


class Runner:
    def __init__(self, opt):
        super(Runner, self).__init__()
        self.opt = opt
        self.generator = torch.nn.DataParallel(Generator().to(opt.device), device_ids=opt.gpu_ids)
        self.discriminator = torch.nn.DataParallel(Discriminator().to(opt.device), device_ids=opt.gpu_ids)
        self.display_id = opt.display_id
        self.vis = Visdom(env='%s' % self.display_id+'_'+str(opt.gpu_ids[0]))

        self.loss_gp = GradientPenalty(lambdaGP=10, gamma=1, device=opt.device)
        self.optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=opt.lr, betas=(0.5, 0.999))

        self.all_sample = []
        self.all_ref = []
        self.autoencoder = torch.nn.DataParallel(Autoencoder(opt).to(opt.device), device_ids=opt.gpu_ids)
        self.metrics = {'1-NN-CD-acc': {'num': 0, 'value': 0},
                        '1-NN-EMD-acc': {'num': 0, 'value': 0},
                        'lgan_cov-CD': {'num': 0, 'value': 0},
                        'lgan_cov-EMD': {'num': 0, 'value': 0},
                        'lgan_mmd-CD': {'num': 0, 'value': 0},
                        'lgan_mmd-EMD': {'num': 0, 'value': 0},
                        'JSD': {'num': 0, 'value': 0}}

    def train_model(self, data, args):
        start = time.time()

        foldername = data[0]
        filename = data[1]
        ref = data[2].to(self.opt.device)
        B, N, _ = ref.shape
        z = torch.normal(mean=0.0, std=0.2, size=(B, 128)).float().to(self.opt.device)
        latent_ref = self.autoencoder.module.encoder(ref)

        # update G network
        if args[1] % 5 - 1 == 0:
            self.latent_gen = self.generator(z)
            fake_out = self.discriminator(self.latent_gen)
            self.loss_G = -torch.mean(fake_out)
            self.optimizerG.zero_grad()
            self.loss_G.backward()
            self.optimizerG.step()

        # update D netwwork
        real_out = self.discriminator(latent_ref)
        fake_out = self.discriminator(self.latent_gen.detach())
        self.loss_D_real = -torch.mean(real_out)
        self.loss_D_fake = torch.mean(fake_out)
        self.loss_D_gp = self.loss_gp(self.discriminator, latent_ref, self.latent_gen.detach())
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_gp
        self.optimizerD.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizerD.step()

        recon = self.autoencoder.module.decoder(latent_ref)
        gen = self.autoencoder.module.decoder(self.latent_gen)

        end = time.time()
        if args[1] % 10 == 0:
            print('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]), end=' ')
            print('Loss: G: %.6f D: %.6f Time: %.6f' % (self.loss_G, self.loss_D, end - start))
            with open(os.path.join(self.opt.save_path, 'runlog.txt'), 'a') as f:
                f.write('[%d/%d][%d/%d]' % (args[0], self.opt.n_epochs, args[1], args[2]))
                f.write('Loss: G: %.6f D: %.6f Time: %.6f\n' % (self.loss_G, self.loss_D, end - start))

            if self.opt.visual:
                plot_diff_pcds([ref[i] for i in range(5)]+[recon[i] for i in range(5)]+[gen[i] for i in range(5)],
                                vis=self.vis,
                                title='train for epoch %d' % args[0],
                                legend=['ref'+str(i) for i in range(5)]+['recon'+str(i) for i in range(5)]+['gen'+str(i) for i in range(5)],
                                win='train_vis')
        
    def gen_sample_from_gaussian(self, data):
        with torch.no_grad():
            ref = data[2].to(self.opt.device)
            B, N, _ = ref.shape
            z = torch.normal(mean=0.0, std=0.2, size=(B, 128)).float().to(self.opt.device)

            self.generator.eval()
            latent_gen = self.generator(z)
            gen = self.autoencoder.module.decoder(latent_gen)

            self.all_sample.append(gen)
            self.all_ref.append(ref)

    def val_model(self, args):
        sample_pcs = torch.cat(self.all_sample, dim=0)
        ref_pcs = torch.cat(self.all_ref, dim=0)

        sample_pcl_npy = sample_pcs.cpu().detach().numpy()
        ref_pcl_npy = ref_pcs.cpu().detach().numpy()
        jsd = JSD(sample_pcl_npy, ref_pcl_npy)
        print("JSD: %s" % jsd)
        with open(os.path.join(self.opt.save_path, 'runlog.txt'), 'a') as f:
            f.write("JSD: %s\n" % jsd)

        if self.opt.visual:
            plot_diff_pcds([ref_pcs[i] for i in range(5)]+[sample_pcs[i] for i in range(5)],
                            vis=self.vis,
                            title='val for epoch %d' % args[0],
                            legend=['ref'+str(i) for i in range(5)]+['gen'+str(i) for i in range(5)],
                            win='val_vis')
        self.all_sample = []
        self.all_ref = []

    def after_one_epoch(self, args):
        self.epoch = args[0]

        losses = {'loss_G': self.loss_G.item(), 'loss_D': self.loss_D.item()}
        plot_loss_curves(self, losses, vis=self.vis, win='loss_curves')

        if self.epoch % 50 == 0:
            save_fn = 'epoch_%d.pth' % (self.epoch)
            save_dir = os.path.join(self.opt.save_path, 'checkpoints')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save({'epoch': self.epoch, 'state_dict': self.generator.state_dict()}, os.path.join(save_dir, save_fn))

    def load_pretrained(self):
        ckpt = torch.load(self.opt.ckpt_path, map_location=lambda storage, location: storage)
        print("load epoch: %d" % (ckpt['epoch']))
        self.generator.load_state_dict(ckpt['state_dict'])

    def load_pretrained_AE(self):
        ckpt = torch.load(self.opt.ae_ckpt_path, map_location=lambda storage, location: storage)
        print("load epoch (pretrained AE): %d" % (ckpt['epoch']))
        self.autoencoder.load_state_dict(ckpt['state_dict'])

    def evaluate_gen(self):
        sample_pcs = torch.cat(self.all_sample, dim=0)
        ref_pcs = torch.cat(self.all_ref, dim=0)
        print("Generation sample size:%s reference size: %s"
            % (sample_pcs.size(), ref_pcs.size()))

        # Save the generative output
        np.save(os.path.join(self.opt.save_path, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy())
        np.save(os.path.join(self.opt.save_path, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

        # Compute metrics
        print("Compute metrics...")
        results = compute_all_metrics(sample_pcs, ref_pcs, self.opt.batch_size, accelerated_cd=True)
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in results.items()}
        pprint(results)

        sample_pcl_npy = sample_pcs.cpu().detach().numpy()
        ref_pcl_npy = ref_pcs.cpu().detach().numpy()
        jsd = JSD(sample_pcl_npy, ref_pcl_npy)
        print("JSD: %s" % jsd)

        self.all_sample = []
        self.all_ref = []
        
        for key, value in results.items():
            if key in self.metrics.keys():
                self.metrics[key]['value'] += value
                self.metrics[key]['num'] += 1
        self.metrics['JSD']['value'] += jsd
        self.metrics['JSD']['num'] += 1
        
    def print_metrics(self):
        for key in self.metrics:
            self.metrics[key]['value'] /= max(self.metrics[key]['num'], 1)
        pprint(self.metrics)
        with open(os.path.join(self.opt.save_path, 'metrics.txt'), 'a') as f:
            f.write(pformat(self.metrics)+'\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=list, default=[0], help='gpu_ids seperated by comma')
    parser.add_argument('--mode', type=str, default='train', help='train | test')
    parser.add_argument('--method', type=str, default='original', help='original | axform')
    parser.add_argument('--class_choice', default=['airplane'], help='category names | None')
    parser.add_argument('--visual', type=bool, default=True, help='visualization during training')
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoints')
    parser.add_argument('--ae_ckpt_path', type=str, help='path to autoencoder checkpoints')

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=2000, help='the epoch number of training the model')
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
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=False,
                                                num_workers=opt.nThreads)

    if opt.mode == "train":
        now = (datetime.utcnow()+timedelta(hours=8)).isoformat()
        opt.save_path = os.path.join('./log', 'latent_3d_points/l-gan', now)
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        os.system('cp ./models/latent_3d_points/l-gan.py %s' % opt.save_path)

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
        runner.load_pretrained_AE()
        for epoch in range(1, opt.n_epochs+1):
            for i, data in enumerate(dataloader):
                runner.train_model(data=data, args=[epoch, i+1, len(dataloader)])
            if epoch % 50 == 0:
                for i, data in enumerate(val_dataloader):
                    runner.gen_sample_from_gaussian(data=data)
                runner.val_model(args=[epoch])
            runner.after_one_epoch(args=[epoch])

    if opt.mode == "test":
        opt.save_path = opt.ckpt_path[::-1].split('/', 3)[-1][::-1]

        runner = Runner(opt)
        runner.load_pretrained_AE()
        runner.load_pretrained()
        for k in range(3):
            for i, data in enumerate(val_dataloader):
                runner.gen_sample_from_gaussian(data=data)
            runner.evaluate_gen()
        runner.print_metrics()