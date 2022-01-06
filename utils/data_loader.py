import torch
import torch.utils.data as data
import os
import math
import json
import h5py
import numpy as np
import open3d as o3d
import randpartial
from visdom import Visdom

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


# ----------------------------------------------------------------------- #
# PCN Dataset
# ----------------------------------------------------------------------- #

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]


class PCNDataset(data.Dataset):
    def __init__(self, root=None, class_choice=None, split='train'):
        """
        It uses all eight partials.

        plane	02691156 | 3795
        cabinet	02933112 | 1322
        car	02958343 | 5677
        chair	03001627 | 5750
        lamp	03636649 | 2068
        sofa	04256520 | 2923
        table	04379243 | 5750
        vessel	04530566 | 1689
        """
        if split == 'train':
            self.partial_num = 8
        else:
            self.partial_num = 1
        
        self.cat = {}
        self.datapath = []  # [(02691156, 1a04e3eab45ca15dd86060f189eb133_0x, xxx.pcd, ...), ...]

        if class_choice is None:
            class_choice = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
        self.cat = {k: v for k, v in cate_to_synsetid.items() if k in class_choice}
        
        with open(os.path.join(root, 'objectid.json'), 'r') as f:
            objectid = json.load(f)[split]
            for item in objectid:
                fn = item.strip().split('/')
                if fn[0] in self.cat.values():
                    for i in range(self.partial_num):
                        partial_path = os.path.join(root, split, 'partial', item.strip(), '0%d.pcd' % (i))
                        gt_path = os.path.join(root, split, 'gt', item.strip()+'.pcd')
                        self.datapath.append((fn[0], fn[1]+'_0%d'%(i), partial_path, gt_path))
    
    def __getitem__(self, index):
        dp = self.datapath[index]
        foldername = dp[0]
        filename = dp[1]
        pcd = o3d.io.read_point_cloud(dp[2])
        partial = np.asarray(pcd.points)
        pcd = o3d.io.read_point_cloud(dp[3])
        gt = np.asarray(pcd.points)

        partial = resample_pcd(partial, 2048)
        # gt = resample_pcd(gt, 2048)

        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(gt).float()

        return foldername, filename, partial, gt
    
    def __len__(self):
        return len(self.datapath)


# ----------------------------------------------------------------------- #
# ShapeNetCore.v2.PC2048 Dataset
# ----------------------------------------------------------------------- #

class ShapeNetCorev2PC2048Dataset(data.Dataset):
    def __init__(self, root=None, class_choice=None, split='train'):
        """
        plane 02691156 | 2832/405/808 4045
        car 02958343 | 2458/352/704 3514
        chair 03001627 | 4612/662/1317 6591
        """
        self.cat = {}
        self.datapath = []  # [(02691156, 1a04e3eab45ca15dd86060f189eb133, xxx.h5), ...]
        
        if class_choice is None:
            class_choice = ['airplane', 'car', 'chair']
        self.cat = {k: v for k, v in cate_to_synsetid.items() if k in class_choice}

        for k, value in self.cat.items():
            foldername = value
            path = os.path.join(root, foldername, split)
            for path, dir_list, file_list in os.walk(path):
                for file_name in file_list:
                    self.datapath.append((foldername, file_name.split('.')[0], os.path.join(path, file_name)))

    def __getitem__(self, index):
        dp = self.datapath[index]
        foldername = dp[0]
        filename = dp[1]
        with h5py.File(dp[2], 'r') as f:
            data = torch.from_numpy(np.array(f['data'])).float()

        return foldername, filename, data

    def __len__(self):
        return len(self.datapath)


if __name__ == "__main__":
    # d = PCNDataset(root='../data/PCN/ShapeNet', class_choice=None, split='train', p_index=0, p_size=1)
    # print(len(d))
    # print(d[0][2], d[0][2].shape)
    # print(d[0][3], d[0][3].shape)

    d = ShapeNetCorev2PC2048Dataset(root='../data/ShapeNetCore.v2.PC2048', class_choice='airplane', split='train')
    print(len(d))
    print(d[0][0])
    print(d[0][1])
    print(d[0][2], d[0][2].shape)