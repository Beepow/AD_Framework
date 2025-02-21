from __future__ import absolute_import, print_function
import os
import utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import data
import scipy.io as sio
from options.testing_options import TestOptions
import utils
import time
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
from memae_3dconv_short import AutoEncoderCov3DMem
from memae_2dconv import AutoEncoderCov2DMem
from datasets2D import Dataset2d
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

###
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
print("Use CUDA : ", use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

###
batch_size_in = opt.BatchSize #1
chnum_in_ = opt.ImgChnNum      # channel number of the input images
framenum_in_ = opt.FrameNum  # frame number of the input images in a video clip
mem_dim_in = opt.MemDim
sparse_shrink_thres = opt.ShrinkThres

img_crop_size = 0

######
model_setting = utils.get_model_setting(opt)

## data path
data_root = opt.DataRoot + f'{opt.Dataset}_T/'


############ model path
model_root = opt.ModelRoot
if(opt.ModelFilePath):
    model_path = opt.ModelFilePath
else:
    model_path = os.path.join(model_root, model_setting + '.pt')

### test result path
te_res_root = opt.OutRoot
te_res_path = te_res_root + 'res_' + model_setting
utils.mkdir(te_res_path)

###### loading trained model
if (opt.ModelName == 'AE'):
    model = AutoEncoderCov3D(chnum_in_)
elif(opt.ModelName=='MemAE'):
    model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
    # model = AutoEncoderCov2DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong Name.')

##
model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

##
if(chnum_in_==1):
    norm_mean = [0.5]
    norm_std = [0.5]
elif(chnum_in_==3):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

if opt.ReSize:
    frame_trans = transforms.Compose([
        transforms.Resize((opt.ReSize, opt.ReSize)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
else:
    frame_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

# ##
# video_list = utils.get_subdir_list(data_root)
video_list = utils.get_file_list(data_root)
video_num = len(video_list)

##
with torch.no_grad():
    error_for_fig = []
    for ite_vid in range(video_num):
        feature_vec = []
        video_name = video_list[ite_vid]
        # video_dataset = data.VideoDatasetOneDir(video_idx_path, video_frame_path, transform=frame_trans)
        video_dataset = data.VideoDataset(data_root=data_root, vid_name=video_name, transform=frame_trans, Test=True)
        video_data_loader = DataLoader(video_dataset,
                                       batch_size=batch_size_in,
                                       shuffle=False
                                       )
        # video_dataset = Dataset2d(data_root=data_root, vid_name=video_name, transform=frame_trans, Test=True)
        # video_data_loader = DataLoader(video_dataset,
        #                                batch_size=batch_size_in,
        #                                shuffle=False
        #                                )

        print('[vidx %02d/%d] [vname %s]' % (ite_vid+1, video_num, video_name))
        recon_error_list = []
        #
        for batch_idx, (item, frames) in enumerate(video_data_loader):
            frames = frames.to(device)
            if (opt.ModelName == 'AE'):
                t0 = time.time()
                recon_frames, feature_vector = model(frames)
                feature_vec.append(feature_vector.cpu().detach().numpy())
                recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
                input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
                r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
                # recon_error = np.mean(sum(r**2)**0.5)
                recon_error = np.mean(r ** 2)  # **0.5
                recon_error_list += [recon_error]
            elif (opt.ModelName == 'MemAE'):
                recon_res, feature_vector = model(frames)
                feature_vec.append(feature_vector.cpu().detach().numpy())
                recon_frames = recon_res['output']
                r = recon_frames - frames
                r = utils.crop_image(r, img_crop_size)
                sp_error_map = torch.sum(r**2, dim=1)**0.5
                s = sp_error_map.size()
                sp_error_vec = sp_error_map.view(s[0], -1)
                recon_error = torch.mean(sp_error_vec, dim=-1)
                recon_error_list += recon_error.cpu().tolist()
            else:
                recon_error = -1
                print('Wrong ModelName.')
        np.save(os.path.join(te_res_path, video_name[:-4] + '.npy'), recon_error_list)
        error_for_fig.append(recon_error_list)

        with open(os.path.join('./results/', 'res.txt'),
                  'a') as RCerr:
            RCerr.write(f'{te_res_path}_{video_name[:-4]}\n')
            formatted_errors = [f'{error:.4f}%' for error in recon_error_list]
            formatted_error_list_str = ', '.join(formatted_errors)
            avg_error = sum(recon_error_list) / len(recon_error_list)
            RCerr.write(f'RCerr : {formatted_error_list_str}\n')
            RCerr.write(f'Avg : {avg_error:.4f}\n')
            RCerr.write(f'Max : {max(recon_error_list):.4f}\n')
            RCerr.write(f'Min : {min(recon_error_list):.4f}\n')

        if not os.path.exists(f"./Feature_Test/Feature_{opt.Dataset}"):
            os.makedirs(f"./Feature_Test/Feature_{opt.Dataset}")
        with open(f"./Feature_Test/Feature_{opt.Dataset}/{video_name[:-4]}", "wb") as f:
            pickle.dump(feature_vec, f, protocol=pickle.HIGHEST_PROTOCOL)

if not os.path.exists(f"./Plot/{opt.Dataset}"):
    os.makedirs(f"./Plot/{opt.Dataset}")
plt.figure(figsize=(4, 4))
for i, errors in enumerate(error_for_fig):
    length = len(errors)
    color = 'r' if i < 20 else 'b'
    # plt.plot(range(total_length, total_length + length), errors, linestyle='-', color=color) #colors[i % len(colors)]
    # total_length += length

    # plt.figure(figsize=(8, 4))
    plt.plot(range(0, length), errors, linestyle='-', color=color)
    plt.title('Reconsturcion Errors in patient level')
    Name = f'{(video_list[i])[:-10]}'
    print(f'{Name} : {np.average(errors)}')
    with open(f"./Plot/{opt.Dataset}/reconstruction_errors.txt", 'a') as f:
        f.write(f'{Name} : {np.average(errors)}\n')
    plt.xlabel(f'{Name}')

    plt.ylabel('Reconstruction Error')
    plt.ylim((0, 1))
    plt.grid(True)

    plt.savefig(f'./Plot/{opt.Dataset}/recon_error_{Name}.png')
    plt.clf()


    over_T = np.where(np.array(errors) >= 0.35)[0]
    print("over 0.35 : ", len(over_T))
    over_35 = np.where(np.array(errors) >= 0.33)[0]
    print("over_33 : ", len(over_35))

    with open(f"./dataset/{opt.Dataset}_T/{Name}_patch.pkl" if i < 20 else f"./dataset/64_patch_T/{Name}_patch.pkl", "rb") as ff: #
        data = pickle.load(ff)
    if len(over_T):
        for idx in over_T:
            selected_data = data[idx]
            slice_3 = selected_data[:, :, 3]  # 7번째 슬라이스
            slice_4 = selected_data[:, :, 4]  # 8번째 슬라이스

            plt.imsave(f"./Plot/{opt.Dataset}/{Name}_{i}_slice_3.png", slice_3, cmap='gray')
            plt.imsave(f"./Plot/{opt.Dataset}/{Name}_{i}_slice_4.png", slice_4, cmap='gray')
            plt.clf()
    else:
        pass


