import os
import argparse
from tqdm import tqdm
import lpips
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
import cv2
from data_RGB import get_test_fivek_data
from abla_exp.CSBNet import CSBNet
import numpy as np
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet_1')

parser.add_argument('--input_dir', default='./Datasets/', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/nfs/project/LLproject/MPRNet-main/Denoising/checkpoints/Denoising/results/MPRNet/abla exp/CSFF/model_psnr_best_24.6380_0.8670.pth', type=str, help='Path to weights')
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

loss_fn_vgg = lpips.LPIPS(net='alex').cuda()

datasets = ['LOLdataset']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'eval')
    test_dataset = get_test_fivek_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    psnr_val_rgb_sys = []
    ssim_val_rgb_sys = []
    lpips_loss = []
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            print("==============={}==================".format(ii))
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            target = data_test[0].cuda()
            input_ = data_test[1].cuda()
            size = data_test[2]
            filenames = data_test[3]

            restored = model_restoration(input_)
            restored = torch.clamp(restored[0], 0, 1) * 255.0
            target = target * 255.0

            lpips_loss.append(loss_fn_vgg(restored, target).cpu().numpy())

            res = restored.data[0].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            tar = target.data[0].squeeze(0).cpu().numpy().transpose(1, 2, 0)

            mean_gray_out = cv2.cvtColor(res.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            mean_gray_gt = cv2.cvtColor(tar.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            cropped_sr_img_adjust = res * (mean_gray_gt / mean_gray_out)

            psnr_val_rgb_sys.append(compare_psnr(tar, cropped_sr_img_adjust, data_range=255.0))
            ssim_val_rgb_sys.append(compare_ssim(tar, cropped_sr_img_adjust, gaussian_weights=True, multichannel=True, data_range=255.0))

            print("test RGB channel sys-------PSNR: {}".format(compare_psnr(tar, cropped_sr_img_adjust, data_range=255.0)))
            print("test RGB channel sys-------SSIM: {}".format(
                compare_ssim(tar, cropped_sr_img_adjust, gaussian_weights=True, multichannel=True, data_range=255.0)))

            # for batch in range(len(restored)):
            #     restored_img = res
            #     utils.save_img_resize((os.path.join(result_dir, filenames[batch] + '.png')), restored_img, size)

        lpips_loss = np.stack(lpips_loss).mean()
        psnr_val_rgb_sys = np.stack(psnr_val_rgb_sys).mean()
        ssim_val_rgb_sys = np.stack(ssim_val_rgb_sys).mean()
    print("test RGB channel-------PSNR: {}  SSIM: {}  LPIPS: {}".format(psnr_val_rgb_sys, ssim_val_rgb_sys, lpips_loss))
    # print("test RGB channel-------PSNR: {}  SSIM: {} ".format(psnr_val_rgb_sys, ssim_val_rgb_sys))
