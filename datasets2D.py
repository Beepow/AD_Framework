from __future__ import print_function, absolute_import
import torch
from PIL import Image
from torch.utils.data import Dataset
import os, os.path
import scipy.io as sio
from skimage import io
from torchvision import transforms
import numpy as np
import pickle
import torchvision.transforms.functional as F

# Video index files are organized in correlated sub folders.
# N x C x T x H x W
class Dataset2d(Dataset):
    def __len__(self):
        # return self.idx_num
        return self.video_len

    def __init__(self, data_root=None, vid_name=None, use_cuda=False, transform=None, Test=False):
        self.Test=Test
        self.use_cuda = use_cuda
        self.transform = transform
        self.data_root = data_root
        self.video_list = [name for name in os.listdir(self.data_root)]
        if not self.Test:
            with open(f"{self.data_root + self.video_list[0]}", "rb") as f:
                self.video_len = (pickle.load(f).shape[0])
        self.vid_name = vid_name

    def __getitem__(self, item, type=None):
        if not self.Test:
            v_name = self.video_list[0]
            v_dir = self.data_root + v_name

            with open(f"{v_dir}", "rb") as f:
                out = pickle.load(f).astype(np.float32)[item]

            return item, out
        # if self.Test==False:
        #     out = []
        #     for i, v_name in enumerate(self.video_list):
        #         v_dir = self.data_root + v_name
        #         with open(f"{v_dir}", "rb") as f:
        #             out.append(pickle.load(f).astype(np.float32))
        #     # Branch = np.array(Branch).astype(np.float32)
        #     out = np.concatenate(out, axis=0)
        #
        #     return item, out

        elif self.Test==True:
            with open(f"{self.data_root + self.vid_name}", "rb") as f:
                Branch = pickle.load(f)
            Branch = Branch.astype(np.float32)

            # Branch_T = torch.cat([self.transform(np.expand_dims(Branch[item][s], axis=2)).unsqueeze(1)
            #                       for s in range (Branch[item].shape[0])],1)
            Branch_T = torch.cat([self.transform(Image.fromarray(Branch[item][s].astype(np.uint8)).convert('L')).unsqueeze(1)
                                  for s in range (Branch[item].shape[0])],1)
            Branch_T = Branch_T.squeeze(-1)
            return item, Branch_T


###
# All video index files are in one dir.
# N x C x T x H x W
class VideoDatasetOneDir(Dataset):
    def __init__(self, idx_dir, frame_root, is_testing=False, use_cuda=False, transform=None):
        self.idx_dir = idx_dir
        self.frame_root = frame_root
        self.idx_name_list = [name for name in os.listdir(self.idx_dir) \
                              if os.path.isfile(os.path.join(self.idx_dir, name))]
        self.idx_name_list.sort()
        self.use_cuda = use_cuda
        self.transform = transform
        self.is_testing = is_testing

    def __len__(self):
        return len(self.idx_name_list)

    def __getitem__(self, item):
        """ get a video clip with stacked frames indexed by the (idx) """
        idx_name = self.idx_name_list[item]
        idx_data = sio.loadmat(os.path.join(self.idx_dir, idx_name))
        v_name = idx_data['v_name'][0]  # video name
        frame_idx = idx_data['idx'][0, :]  # frame index list for a video clip
        v_dir = self.frame_root
        #
        tmp_frame = io.imread(os.path.join(v_dir, ('%03d' % frame_idx[0]) + '.jpg'))

        tmp_frame_shape = tmp_frame.shape
        frame_cha_num = len(tmp_frame_shape)
        h = tmp_frame_shape[0]
        w = tmp_frame_shape[1]
        if frame_cha_num == 3:
            c = tmp_frame_shape[2]
        elif frame_cha_num == 2:
            c = 1
        # each sample is concatenation of the indexed frames
        if self.transform:
            # frames = torch.cat([self.transform(
            #     io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg')).reshape(h, w, c)).resize_(c, 1, h, w) for i
            #                     in frame_idx], 1)
            frames = torch.cat([self.transform(
                Image.fromarray(io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg'))).convert("L")).unsqueeze(1)
                                for i
                                in frame_idx], 1)
        else:
            tmp_frame_trans = transforms.ToTensor()  # trans Tensor
            frames = torch.cat([self.transform(
                io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg')).reshape(h, w, c)).resize_(c, 1, h, w) for i
                                in frame_idx], 1)

        return item, frames
