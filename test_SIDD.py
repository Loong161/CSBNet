import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
import time
from data_RGB import get_test_data
from abla_exp.CSBNet import CSBNet
import numpy as np
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet_1')

parser.add_argument('--input_dir', default='./Datasets/', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/nfs/project/LLproject/MPRNet-main/Denoising/checkpoints/Denoising/results/fiveK/3/model_psnr_best_25.8946_0.9179.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = CSBNet()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

datasets = ['VV']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset)
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    avg_time = []
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            print("==============={}==================".format(ii))
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # target = data_test[0]
            input_ = data_test[0].cuda()

            filenames = data_test[1]
            size = data_test[2]

            bt = time.time()
            restored = model_restoration(input_)
            et = time.time()
            avg_time.append(et - bt)
            restored = torch.clamp(restored[0], 0, 1) * 255.0
            # target = target * 255.0

            res = restored.data[0].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            # tar = target.data[0].squeeze(0).cpu().numpy().transpose(1, 2, 0)

            # psnr_val_rgb_sys.append(compare_psnr(tar, res, data_range=255.0))
            # ssim_val_rgb_sys.append(compare_ssim(tar, res, gaussian_weights=True, multichannel=True, data_range=255.0))

            # print("test RGB channel sys-------PSNR: {}".format(compare_psnr(tar, res, data_range=255.0)))
            # print("test RGB channel sys-------SSIM: {}".format(
            #     compare_ssim(tar, res, gaussian_weights=True, multichannel=True, data_range=255.0)))

            for batch in range(len(restored)):
                restored_img = res
                utils.save_img_resize((os.path.join(result_dir, filenames[batch] + '.png')), restored_img, size)

        # psnr_val_rgb_sys = np.stack(psnr_val_rgb_sys).mean()
        # ssim_val_rgb_sys = np.stack(ssim_val_rgb_sys).mean()
        avg_time = np.stack(avg_time).mean()
    print("test time: {} ".format(avg_time))
