from __future__ import absolute_import, print_function
import torch
from torch import nn

from models import MemModule

class AutoEncoderCov2DMem(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres=0.0025):
        super(AutoEncoderCov2DMem, self).__init__()
        print('AutoEncoderCov2DMem')
        self.chnum_in = chnum_in
        feature_num = 64      #128 / 4
        feature_num_2 = 32    #96 / 3
        feature_num_x2 = 128   #256 / 8
        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in, feature_num_2, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2, feature_num, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=feature_num_x2, shrink_thres =shrink_thres)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_num_x2, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_x2, feature_num, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num, feature_num_2, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_2, self.chnum_in, (3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        feature_vec = self.encoder(x)
        res_mem = self.mem_rep(feature_vec)
        feature_mem = res_mem['output']
        att = res_mem['att']
        output = self.decoder(feature_mem)
        return {'output': output, 'att': att}, feature_vec
