import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
from pdb import set_trace as stx
import random

def shapness_measure(img_temp, kernel_size):
    conv_x = cv2.Sobel(img_temp,cv2.CV_64F,1,0,ksize=kernel_size)
    conv_y = cv2.Sobel(img_temp,cv2.CV_64F,0,1,ksize=kernel_size)
    temp_arr_x = deepcopy(conv_x*conv_x)
    temp_arr_y = deepcopy(conv_y*conv_y)
    temp_sum_x_y = temp_arr_x+temp_arr_y
    temp_sum_x_y = np.sqrt(temp_sum_x_y)
    return np.sum(temp_sum_x_y)

def filter_patch_sharpness(patches_src_temp, patches_trg_temp):
    patches_src, patches_trg = [], []
    fitnessVal_3 = []
    fitnessVal_7 = []
    fitnessVal_11 = []
    fitnessVal_15 = []
    num_of_img_patches = len(patches_trg_temp)
    
    for i in range(num_of_img_patches):
        fitnessVal_3.append(shapness_measure(cv2.cvtColor(patches_trg_temp[i], cv2.COLOR_BGR2GRAY),3))
        fitnessVal_7.append(shapness_measure(cv2.cvtColor(patches_trg_temp[i], cv2.COLOR_BGR2GRAY),7))
        fitnessVal_11.append(shapness_measure(cv2.cvtColor(patches_trg_temp[i], cv2.COLOR_BGR2GRAY),11))
        fitnessVal_15.append(shapness_measure(cv2.cvtColor(patches_trg_temp[i], cv2.COLOR_BGR2GRAY),15))
    
    fitnessVal_3 = np.asarray(fitnessVal_3)
    fitnessVal_7 = np.asarray(fitnessVal_7)
    fitnessVal_11 = np.asarray(fitnessVal_11)
    fitnessVal_15 = np.asarray(fitnessVal_15)
    
    fitnessVal_3 = (fitnessVal_3-np.min(fitnessVal_3))/np.max((fitnessVal_3-np.min(fitnessVal_3)))
    fitnessVal_7 = (fitnessVal_7-np.min(fitnessVal_7))/np.max((fitnessVal_7-np.min(fitnessVal_7)))
    fitnessVal_11 = (fitnessVal_11-np.min(fitnessVal_11))/np.max((fitnessVal_11-np.min(fitnessVal_11)))
    fitnessVal_15 = (fitnessVal_15-np.min(fitnessVal_15))/np.max((fitnessVal_15-np.min(fitnessVal_15)))
    
    fitnessVal_all = fitnessVal_3*fitnessVal_7*fitnessVal_11*fitnessVal_15
    
    to_remove_patches_number = int(to_remove_ratio*num_of_img_patches)
    
    for itr in range(to_remove_patches_number):
        minArrInd = np.argmin(fitnessVal_all)
        fitnessVal_all[minArrInd] = 2
    
    for itr in range(num_of_img_patches):
        if fitnessVal_all[itr] != 2:
            patches_src.append(patches_src_temp[itr])
            patches_trg.append(patches_trg_temp[itr])
    
    return patches_src, patches_trg

def slice_stride(_img_src, _img_trg):
    coordinates_list = []
    coordinates_list.append([0,0,0,0])
    patches_src_temp, patches_trg_temp = [], []
    
    for r in range(0,_img_src.shape[0],stride[0]):
        for c in range(0,_img_src.shape[1],stride[1]):
            if (r+patch_size[0]) <= _img_src.shape[0] and (c+patch_size[1]) <= _img_src.shape[1]:
                patches_src_temp.append(_img_src[r:r+patch_size[0],c:c+patch_size[1]])
                patches_trg_temp.append(_img_trg[r:r+patch_size[0],c:c+patch_size[1]])

            elif (r+patch_size[0]) <= _img_src.shape[0] and not ([r,r+patch_size[0],_img_src.shape[1]-patch_size[1],_img_src.shape[1]] in coordinates_list):
                patches_src_temp.append(_img_src[r:r+patch_size[0],_img_src.shape[1]-patch_size[1]:_img_src.shape[1]])
                patches_trg_temp.append(_img_trg[r:r+patch_size[0],_img_trg.shape[1]-patch_size[1]:_img_trg.shape[1]])
                coordinates_list.append([r,r+patch_size[0],_img_src.shape[1]-patch_size[1],_img_src.shape[1]])
                
            elif (c+patch_size[1]) <= _img_src.shape[1] and not ([_img_src.shape[0]-patch_size[0],_img_src.shape[0],c,c+patch_size[1]] in coordinates_list):
                patches_src_temp.append(_img_src[_img_src.shape[0]-patch_size[0]:_img_src.shape[0],c:c+patch_size[1]])
                patches_trg_temp.append(_img_trg[_img_trg.shape[0]-patch_size[0]:_img_trg.shape[0],c:c+patch_size[1]])
                coordinates_list.append([_img_src.shape[0]-patch_size[0],_img_src.shape[0],c,c+patch_size[1]])
                
            elif not ([_img_src.shape[0]-patch_size[0],_img_src.shape[0],_img_src.shape[1]-patch_size[1],_img_src.shape[1]] in coordinates_list):
                patches_src_temp.append(_img_src[_img_src.shape[0]-patch_size[0]:_img_src.shape[0],_img_src.shape[1]-patch_size[1]:_img_src.shape[1]])
                patches_trg_temp.append(_img_trg[_img_trg.shape[0]-patch_size[0]:_img_trg.shape[0],_img_trg.shape[1]-patch_size[1]:_img_trg.shape[1]])
                coordinates_list.append([_img_src.shape[0]-patch_size[0],_img_src.shape[0],_img_src.shape[1]-patch_size[1],_img_src.shape[1]])

    return patches_src_temp, patches_trg_temp

def process_image_pair(file_, is_val=False):
    src_file, trg_file = file_
    filename = os.path.splitext(os.path.split(src_file)[-1])[0]
    
    src_img = cv2.imread(src_file, -1)
    trg_img = cv2.imread(trg_file, -1)

    if is_val:
        # For validation, take center crop
        h, w = src_img.shape[:2]
        i = (h - val_patch_size) // 2
        j = (w - val_patch_size) // 2
        
        src_patch = src_img[i:i + val_patch_size, j:j + val_patch_size]
        trg_patch = trg_img[i:i + val_patch_size, j:j + val_patch_size]
        
        src_savename = os.path.join(src_val_tar, filename + '.png')
        trg_savename = os.path.join(trg_val_tar, filename + '.png')
        
        cv2.imwrite(src_savename, src_patch)
        cv2.imwrite(trg_savename, trg_patch)
    else:
        # For training, extract and filter patches
        src_patches, trg_patches = slice_stride(src_img, trg_img)
        src_patches, trg_patches = filter_patch_sharpness(src_patches, trg_patches)
        
        num_patch = 0
        for src_patch, trg_patch in zip(src_patches, trg_patches):
            num_patch += 1
            
            src_savename = os.path.join(src_train_tar, filename + '-' + str(num_patch) + '.png')
            trg_savename = os.path.join(trg_train_tar, filename + '-' + str(num_patch) + '.png')
            
            cv2.imwrite(src_savename, src_patch)
            cv2.imwrite(trg_savename, trg_patch)

# Parameters
patch_size = [512, 512]
stride = [204, 204]
val_patch_size = 256
p_max = 0
to_remove_ratio = 0.3
num_cores = -1
train_ratio = 0.85  # 85% for training

# Set random seed for reproducibility
random.seed(42)

# Paths
src = './defocus_formal'  # Change this to your dataset path
tar = './Datasets/LSD'  # Change this to your output path

# Create directories
src_train_tar = os.path.join(tar, 'train/input_crops')
trg_train_tar = os.path.join(tar, 'train/target_crops')
src_val_tar = os.path.join(tar, 'val/input_crops')
trg_val_tar = os.path.join(tar, 'val/target_crops')

os.makedirs(src_train_tar, exist_ok=True)
os.makedirs(trg_train_tar, exist_ok=True)
os.makedirs(src_val_tar, exist_ok=True)
os.makedirs(trg_val_tar, exist_ok=True)

# Get file lists
src_files = natsorted(glob(os.path.join(src, 'train/blurry', '*.png')))
trg_files = natsorted(glob(os.path.join(src, 'train/sharp', '*.png')))

# Create file pairs
files = list(zip(src_files, trg_files))

# Randomly shuffle the files
random.shuffle(files)

# Split into train and validation
split_idx = int(len(files) * train_ratio)
train_files = files[:split_idx]
val_files = files[split_idx:]

test_files = val_files[500:]

print(f"Total images: {len(files)}")
print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")

test_dir_input = os.path.join(tar, 'test/input_crops')
test_dir_target = os.path.join(tar, 'test/target_crops')

os.makedirs(test_dir_input, exist_ok=True)
os.makedirs(test_dir_target, exist_ok=True)

# Save test files
print(f"Test images: {len(test_files)}")
for src_file, trg_file in tqdm(test_files, desc="Saving test files"):
    filename = os.path.splitext(os.path.split(src_file)[-1])[0]
    
    src_img = cv2.imread(src_file, -1)
    trg_img = cv2.imread(trg_file, -1)
    
    src_savename = os.path.join(test_dir_input, filename + '.png')
    trg_savename = os.path.join(test_dir_target, filename + '.png')
    
    cv2.imwrite(src_savename, src_img)
    cv2.imwrite(trg_savename, trg_img)

print("Test files saved successfully.")

# # Process training files
# print("Processing training files...")
# Parallel(n_jobs=num_cores)(
#     delayed(process_image_pair)(file_, False) for file_ in tqdm(train_files)
# )

# # Process validation files
# print("Processing validation files...")
# Parallel(n_jobs=num_cores)(
#     delayed(process_image_pair)(file_, True) for file_ in tqdm(val_files)
# )

print("Done!")