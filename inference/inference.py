import numpy as np
import os
import argparse
from tqdm import tqdm
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import warnings
import kornia.color
import cv2
import lpips
import logging
import csv
import yaml

warnings.filterwarnings("ignore")

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

# Argument Parser
parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')
parser.add_argument('--input_dir', default='./inference/dataset/motion/testrealblur/RealBlur-R', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./inference/results/motion/testrealblur/RealBlur-R', type=str, help='Directory for results')
parser.add_argument('--weights', default='./inference/models/initial_pretrained/motion/motion_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='RealBlur-R', type=str, help='Test Dataset')  # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

# LPIPS Model
alex = lpips.LPIPS(net='alex').cuda()

# Logging Setup
model_name = os.path.basename(args.weights).replace(".pth", "")
log_dir = f"./inference/logs/{model_name}_{args.dataset}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inference_run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Metrics CSV Setup
csv_file = os.path.join(log_dir, f"metrics_{model_name}_{args.dataset}.csv")
csv_headers = ['Image', 'PSNR', 'SSIM', 'MAE', 'LPIPS', 'DeltaE', 'Category']

yaml_file = './inference/Options/Deblurring_Restormer.yml'

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
logging.info(f"Testing using model: {model_name}, dataset: {args.dataset}")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Directories Setup
factor = 8
dataset = args.dataset
result_dir = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

good_dir = os.path.join(result_dir, "good")
bad_dir = os.path.join(result_dir, "bad")
neutral_dir = os.path.join(result_dir, "neutral")
all_dir = os.path.join(result_dir, "all")
os.makedirs(good_dir, exist_ok=True)
os.makedirs(bad_dir, exist_ok=True)
os.makedirs(neutral_dir, exist_ok=True)
os.makedirs(all_dir, exist_ok=True)

inp_dir = os.path.join(args.input_dir, 'input')
tar_dir = os.path.join(args.input_dir, 'target')

filesI = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
filesC = natsorted(glob(os.path.join(tar_dir, '*.png')) + glob(os.path.join(tar_dir, '*.jpg')))

logging.info(f"Found {len(filesI)} input images and {len(filesC)} target images.")

psnr, mae, ssim, pips, color_diffs = [], [], [], [], []

# Write CSV Header
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# Start Inference
with torch.no_grad():
    try:
        for fileI, fileC in tqdm(zip(filesI, filesC), total=len(filesC)):
            filename = os.path.basename(fileI)
            result_path = os.path.join(all_dir, filename)
            if os.path.exists(result_path):
                logging.info(f"Skipping already processed file: {filename}")
                continue

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            imgI = np.float32(utils.load_img(fileI)) / 255.0
            imgI = cv2.resize(imgI, (1280, 720), interpolation=cv2.INTER_AREA)

            imgC = np.float32(utils.load_img(fileC)) / 255.0
            imgC = cv2.resize(imgC, (1280, 720), interpolation=cv2.INTER_AREA)

            imgI = torch.from_numpy(imgI).permute(2, 0, 1)
            target = torch.from_numpy(imgC).permute(2, 0, 1)
            input_ = imgI.unsqueeze(0).cuda()
            target = target.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1)
            psps = alex(target, restored, normalize=True).item()

            delta_e_cie2000 = delta_e_cie2000_torch(target, restored)
            delta_e_cie2000 = delta_e_cie2000.mean().item()

            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored_uint8 = np.uint8((restored * 255).round())

            # Save the restored image with metrics
            metrics_suffix = f"_PSNR{utils.PSNR(imgC, restored):.2f}_SSIM{utils.SSIM(imgC, restored):.2f}_MAE{utils.MAE(imgC, restored):.2f}_LPIPS{psps:.2f}_DeltaE{delta_e_cie2000:.2f}"
            all_result_path = os.path.join(all_dir, f"{filename}{metrics_suffix}.png")
            utils.save_img(all_result_path, restored_uint8)

            # Categorize results
            psnr_value = utils.PSNR(imgC, restored)
            category = "good" if psnr_value >= 30 else "bad" if psnr_value <= 20 else "neutral"
            category_dir = good_dir if category == "good" else bad_dir if category == "bad" else neutral_dir
            category_result_path = os.path.join(category_dir, f"{filename}{metrics_suffix}.png")
            utils.save_img(category_result_path, restored_uint8)

            # Store metrics in CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, psnr_value, utils.SSIM(imgC, restored), utils.MAE(imgC, restored), psps, delta_e_cie2000, category])

            # Append to summary metrics
            psnr.append(psnr_value)
            mae.append(utils.MAE(imgC, restored))
            ssim.append(utils.SSIM(imgC, restored))
            pips.append(psps)
            color_diffs.append(delta_e_cie2000)

        logging.info("Inference completed successfully.")

    except Exception as e:
        logging.error(f"Error during inference: {e}")

# Log Overall Metrics
psnr, mae, ssim, pips, color_diffs = map(np.array, [psnr, mae, ssim, pips, color_diffs])
logging.info(f"Overall Metrics - PSNR: {np.mean(psnr):.4f}, SSIM: {np.mean(ssim):.4f}, "
             f"MAE: {np.mean(mae):.4f}, LPIPS: {np.mean(pips):.4f}, DeltaE: {np.mean(color_diffs):.4f}")