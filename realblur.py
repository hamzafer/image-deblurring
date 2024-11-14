import os
import shutil

def organize_real_blur_dataset(txt_file_path, base_dir, input_dir, target_dir):
    # Create input and target directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    count = 1
    # Read the text file
    with open(txt_file_path, 'r') as file:
        for line in file:
            # Split the line into blur and gt paths
            gt_path, blur_path = line.strip().split()

            # Construct full paths
            
            full_blur_path = os.path.join(base_dir, blur_path)
            full_gt_path = os.path.join(base_dir, gt_path)

            # Extract filenames
            blur_filename = f"blur_{count}.png"
            gt_filename =  f"gt_{count}.png"

            # Copy files to their respective directories
            shutil.copy(full_blur_path, os.path.join(input_dir, blur_filename))
            shutil.copy(full_gt_path, os.path.join(target_dir, gt_filename))

            count+=1

    print("Dataset organization complete.")

# Example usage
txt_file_path = '/media/user1/Data storage/Akmaral/DL/Restormer/RealBlur_J_train_list.txt'
base_dir = '.'
input_dir = './Motion_Deblurring/Datasets/RealBlur-J/input/'
target_dir = './Motion_Deblurring/Datasets/RealBlur-J/target/'

organize_real_blur_dataset(txt_file_path, base_dir, input_dir, target_dir)