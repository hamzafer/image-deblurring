## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
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


parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/RealBlur-J', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/motion_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='RealBlurJ', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/Deblurring_Restormer.yml'
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


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

inp_dir = os.path.join(args.input_dir, 'target')
tar_dir = os.path.join(args.input_dir, 'input')

filesI = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
filesC = natsorted(glob(os.path.join(tar_dir, '*.png')) + glob(os.path.join(tar_dir, '*.jpg')))

psnr, mae, ssim, pips = [], [], [], []
color_diffs = []

with torch.no_grad():
    for fileI, fileC in tqdm(zip(filesI, filesC), total=len(filesC)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        imgI = np.float32(utils.load_img(fileI))/255.
        imgC = np.float32(utils.load_img(fileC))/255.
        imgI = torch.from_numpy(imgI).permute(2,0,1)
        target = torch.from_numpy(imgC).permute(2,0,1)
        input_ = imgI.unsqueeze(0).cuda()
        target = target.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        #target = F.pad(target, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1)
        psps = alex(target, restored, normalize=True).item()
        pips.append(alex(target, restored, normalize=True).item())

        delta_e_cie2000 = delta_e_cie2000_torch(target, restored)
        delta_e_cie2000 = delta_e_cie2000.mean().item()
        color_diffs.append(delta_e_cie2000)

        restored = restored.cpu().detach().permute(0, 2, 3,1).squeeze(0).numpy()
        

        psnr.append(utils.PSNR(imgC, restored))
        mae.append(utils.MAE(imgC, restored))
        ssim.append(utils.SSIM(imgC, restored))

        if utils.PSNR(imgC, restored) <= 20 :
            print("Image {}: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(fileC.split("/")[-1], (utils.PSNR(imgC, restored)), (utils.SSIM(imgC, restored)), (utils.MAE(imgC, restored)), (psps), delta_e_cie2000))
            save_file = os.path.join(result_dir, "bad", os.path.split(fileC)[-1])
            restored = np.uint8((restored*255).round())
            utils.save_img(save_file, restored)

        if utils.PSNR(imgC, restored) >= 33:
            print("Image {}: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(fileC.split("/")[-1], (utils.PSNR(imgC, restored)), (utils.SSIM(imgC, restored)), (utils.MAE(imgC, restored)), (psps), delta_e_cie2000))
            save_file = os.path.join(result_dir, "good", os.path.split(fileC)[-1])
            restored = np.uint8((restored*255).round())
            utils.save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)
color_diffs = np.array(color_diffs)

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f} Color Delta {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae), np.mean(pips), np.mean(color_diffs)))