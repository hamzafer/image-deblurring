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
from basicsr.models.archs.reversible_restormer_arch import ReversibleRestormer
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
    """
    lab1 = kornia.color.rgb_to_lab(lab1)
    lab2 = kornia.color.rgb_to_lab(lab2)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

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

    L_term = delta_L_prime / (1 + 0.015 * (L_avg - 50) ** 2)
    C_term = delta_C_prime / (1 + 0.045 * C_avg_prime)
    H_term = delta_h_prime / (1 + 0.015 * C_avg_prime)

    delta_e = torch.sqrt(L_term**2 + C_term**2 + H_term**2)
    return delta_e

yaml_file = './inference/Options/Reversible_Restomer.yml'

# Arguments
weights_path = './inference/models/improved_from_scratch/best_new_model.pth'
input_dir_path = './inference/dataset/motion/testrealblur/RealBlur-J'
result_base_dir = './inference/results/motion/testrealblur/RealBlur-J'

model_name = os.path.basename(weights_path).replace(".pth", "")
dataset_name = os.path.basename(os.path.normpath(input_dir_path))

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')
parser.add_argument('--input_dir', default=input_dir_path, type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=result_base_dir, type=str, help='Directory for results')
parser.add_argument('--weights', default=weights_path, type=str, help='Path to weights')
parser.add_argument('--dataset', default=f"{model_name}_{dataset_name}", type=str, help='Dataset name derived from model and input directory')
args = parser.parse_args()

# LPIPS Model
alex = lpips.LPIPS(net='alex').cuda()

# Logging Setup
log_dir = f"./inference/logs/{model_name}_{dataset_name}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inference_run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Metrics CSV Setup
csv_file = os.path.join(log_dir, f"metrics_{model_name}_{dataset_name}.csv")
csv_headers = ['Image', 'PSNR', 'SSIM', 'MAE', 'LPIPS', 'DeltaE', 'Category']

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
x['network_g'].pop('type')

# Initialize Model
#model_restoration = Restormer(**x['network_g'])
model_restoration = ReversibleRestormer(**x['network_g'])
checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

logging.info("==========================================")
logging.info(f"Starting inference for model: {model_name}, dataset: {dataset_name}")
logging.info(f"Input Directory: {args.input_dir}")
logging.info(f"Results Directory: {args.result_dir}")
logging.info("==========================================")

# Directory Setup
result_dir = os.path.join(args.result_dir, f"{model_name}_{dataset_name}")
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

# CSV Header
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# Skipping Logic
processed_base_names = set(
    os.path.splitext(os.path.basename(f))[0].split('_PSNR')[0] for f in glob(os.path.join(all_dir, '*.png'))
)

# Metrics
psnr, mae, ssim, pips, color_diffs = [], [], [], [], []

# Inference
start_time = time.time()
with torch.no_grad():
    try:
        for fileI, fileC in tqdm(zip(filesI, filesC), total=len(filesC)):
            filename = os.path.basename(fileI)
            base_name = os.path.splitext(filename)[0]
            if base_name in processed_base_names:
                logging.info(f"Skipping already processed file: {filename}")
                continue

            logging.info(f"Processing file: {filename}")

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            imgI = np.float32(utils.load_img(fileI)) / 255.0
            # imgI = cv2.resize(imgI, (1280, 720), interpolation=cv2.INTER_AREA)

            imgC = np.float32(utils.load_img(fileC)) / 255.0
            # imgC = cv2.resize(imgC, (1280, 720), interpolation=cv2.INTER_AREA)

            imgI = torch.from_numpy(imgI).permute(2, 0, 1).unsqueeze(0).cuda()
            target = torch.from_numpy(imgC).permute(2, 0, 1).unsqueeze(0).cuda()

            h, w = imgI.shape[2], imgI.shape[3]
            H, W = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
            padh, padw = H - h, W - w
            input_ = F.pad(imgI, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)
            restored = restored[:, :, :h, :w].clamp(0, 1)
            restored_np = restored.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            psps = alex(target, restored).item()
            delta_e_cie2000 = delta_e_cie2000_torch(target, restored).mean().item()

            # Metrics
            psnr_value = utils.PSNR(imgC, restored_np)
            category = "good" if psnr_value >= 30 else "bad" if psnr_value <= 20 else "neutral"
            metrics_suffix = f"_PSNR{psnr_value:.2f}_SSIM{utils.SSIM(imgC, restored_np):.2f}_MAE{utils.MAE(imgC, restored_np):.2f}_LPIPS{psps:.2f}_DeltaE{delta_e_cie2000:.2f}"
            result_path = os.path.join(all_dir, f"{base_name}{metrics_suffix}.png")
            utils.save_img(result_path, (restored_np * 255).astype(np.uint8))

            category_dir = good_dir if category == "good" else bad_dir if category == "bad" else neutral_dir
            utils.save_img(os.path.join(category_dir, os.path.basename(result_path)), (restored_np * 255).astype(np.uint8))

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, psnr_value, utils.SSIM(imgC, restored_np), utils.MAE(imgC, restored_np), psps, delta_e_cie2000, category])

            psnr.append(psnr_value)
            mae.append(utils.MAE(imgC, restored_np))
            ssim.append(utils.SSIM(imgC, restored_np))
            pips.append(psps)
            color_diffs.append(delta_e_cie2000)

    except Exception as e:
        logging.error(f"Error during inference: {e}")

end_time = time.time()
logging.info(f"Inference completed in {end_time - start_time:.2f} seconds.")
logging.info(f"Overall Metrics - PSNR: {np.mean(psnr):.4f}, SSIM: {np.mean(ssim):.4f}, "
             f"MAE: {np.mean(mae):.4f}, LPIPS: {np.mean(pips):.4f}, DeltaE: {np.mean(color_diffs):.4f}")
