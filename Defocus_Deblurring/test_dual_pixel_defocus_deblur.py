"""
## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch

from skimage import img_as_ubyte
from basicsr.models.archs.restormer_arch import Restormer
import cv2
import utils
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx
import warnings
warnings.filterwarnings("ignore")

import kornia.color

import lpips
alex = lpips.LPIPS(net='alex').cuda()

def delta_e_cie2000_torch(lab1, lab2):
    """
    Compute CIEDE2000 color difference for tensors on GPU.

    Args:
        lab1, lab2: Tensors of shape (batch_size, height, width, 3) or (height, width, 3)
                    where last dimension represents [L, a, b].

    Returns:
        Delta E (CIEDE2000) color difference as a tensor of shape (batch_size, height, width).
    """
    # Convert tensors to LabColor objects
    lab1 =  kornia.color.rgb_to_lab(lab1)
    lab2 =  kornia.color.rgb_to_lab(lab2)
    
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Mean values
    L_avg = (L1 + L2) / 2.0
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0

    G = 0.5 * (1 - torch.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = torch.sqrt(a1_prime**2 + b1**2)
    C2_prime = torch.sqrt(a2_prime**2 + b2**2)
    C_avg_prime = (C1_prime + C2_prime) / 2.0

    h1_prime = torch.atan2(b1, a1_prime)
    h2_prime = torch.atan2(b2, a2_prime)

    torch.pi = torch.acos(torch.tensor(-1.0))
    h1_prime = h1_prime % (2 * torch.pi)
    h2_prime = h2_prime % (2 * torch.pi)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    h_diff = h2_prime - h1_prime
    delta_h_prime = 2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(h_diff / 2)

    # Weighted terms
    L_term = delta_L_prime / (1 + 0.015 * (L_avg - 50)**2)
    C_term = delta_C_prime / (1 + 0.045 * C_avg_prime)
    H_term = delta_h_prime / (1 + 0.015 * C_avg_prime)

    # Final delta E calculation
    delta_e = torch.sqrt(L_term**2 + C_term**2 + H_term**2)

    return delta_e


parser = argparse.ArgumentParser(description='Dual Pixel Defocus Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/Downloads/DPDD/test', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Dual_Pixel_Defocus_Deblurring/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/dual_pixel_defocus_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/DefocusDeblur_DualPixel_16bit_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesL = natsorted(glob(os.path.join(args.input_dir, 'inputL', '*.png')))
filesR = natsorted(glob(os.path.join(args.input_dir, 'inputR', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))

indoor_labels  = np.load('./Datasets/Downloads/DPDD/test/indoor_labels.npy')
outdoor_labels = np.load('./Datasets/Downloads/DPDD/test/outdoor_labels.npy')

psnr, mae, ssim, pips = [], [], [], []
color_diffs = []

with torch.no_grad():
    for fileL, fileR, fileC in tqdm(zip(filesL, filesR, filesC), total=len(filesC)):

        imgL = np.float32(utils.load_img16(fileL))/65535.
        imgR = np.float32(utils.load_img16(fileR))/65535.
        imgC = np.float32(utils.load_img16(fileC))/65535.

        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0,3,1,2).cuda()
        patchL = torch.from_numpy(imgL).unsqueeze(0).permute(0,3,1,2)
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0,3,1,2)

        input_ = torch.cat([patchL, patchR], 1).cuda()

        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)
        psps = alex(patchC, restored, normalize=True).item()
        pips.append(alex(patchC, restored, normalize=True).item())

        delta_e_cie2000 = delta_e_cie2000_torch(patchC, restored)
        delta_e_cie2000 = delta_e_cie2000.mean().item()
        color_diffs.append(delta_e_cie2000)

        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(utils.PSNR(imgC, restored))
        mae.append(utils.MAE(imgC, restored))
        ssim.append(utils.SSIM(imgC, restored))

         # Convert images to Lab for color difference computation
        imgC_lab = cv2.cvtColor((imgC * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)
        restored_lab = cv2.cvtColor((restored * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)

        # Compute CIEDE2000 for each pixel and average across the image


        if utils.PSNR(imgC, restored) <= 20.5 :
            print("Image {}: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(fileC.split("/")[-1], (utils.PSNR(imgC, restored)), (utils.SSIM(imgC, restored)), (utils.MAE(imgC, restored)), (psps), delta_e_cie2000))
            save_file = os.path.join(result_dir, "bad", os.path.split(fileC)[-1])
            restored = np.uint8((restored*255).round())
            utils.save_img(save_file, restored)

        if utils.PSNR(imgC, restored) >=30:
            print("Image {}: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(fileC.split("/")[-1], (utils.PSNR(imgC, restored)), (utils.SSIM(imgC, restored)), (utils.MAE(imgC, restored)), (psps), delta_e_cie2000))
            save_file = os.path.join(result_dir, "good", os.path.split(fileC)[-1])
            restored = np.uint8((restored*255).round())
            utils.save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)
color_diffs = np.array(color_diffs)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels-1], mae[indoor_labels-1], ssim[indoor_labels-1], pips[indoor_labels-1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels-1], mae[outdoor_labels-1], ssim[outdoor_labels-1], pips[outdoor_labels-1]
color_diffs_indoor = color_diffs[indoor_labels-1]
color_diffs_outdoor = color_diffs[outdoor_labels-1]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae), np.mean(pips), np.mean(color_diffs)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor), np.mean(mae_indoor), np.mean(pips_indoor), np.mean(color_diffs_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor), np.mean(mae_outdoor), np.mean(pips_outdoor), np.mean(color_diffs_outdoor)))
