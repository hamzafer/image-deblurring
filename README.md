# Efficient Transformer for High-Resolution Image Motion Deblurring
<img width="807" alt="image" src="https://github.com/user-attachments/assets/1137b1cf-650e-4e23-8a8b-1afb3d3b206b">

## Overview

This project builds upon the [Restormer](https://github.com/swz30/Restormer) architecture, enhancing its efficiency and performance for the task of high-resolution image motion deblurring. The improvements include architectural modifications, advanced training techniques, and extended evaluations on diverse datasets to create a robust and efficient model for real-world deblurring challenges.

### Key Features of This Work
- **Reduced Model Complexity:** The model complexity is reduced by 18.4%, improving inference speed and reducing memory requirements.
- **Enhanced Training Pipeline:** Incorporation of transformations such as color jitter, Gaussian blur, perspective transforms, and a new frequency-domain loss function to improve robustness and accuracy.
- **Extensive Evaluation:** Experiments performed on RealBlur-R, RealBlur-J, and the Ultra-High-Definition Motion Blurred (UHDM) datasets.
- **Ablation Studies:** Detailed analyses to quantify the impact of architectural and training modifications.

This project retains the core innovations of Restormer, including its multi-Dconv head transposed attention mechanism and gated-Dconv feed-forward network, while introducing custom enhancements tailored to motion deblurring tasks.

## Architectural Modifications

### Key Changes
1. **Reduction in Parameters:** Number of layers and transformer blocks reduced to lower computational overhead.
2. **Increased Attention Heads:** Doubling attention heads per stage to enhance feature extraction while balancing computational costs.
3. **Custom Loss Function:** Integration of a frequency-domain loss alongside L1 pixel-wise loss for better preservation of fine details.
<img width="807" alt="image" src="https://github.com/user-attachments/assets/db34f4a8-8052-470d-8584-fd7869ed6960">

These modifications resulted in faster convergence, improved stability, and better performance across a range of datasets and challenging scenarios.

## Training Enhancements

### Added Transformations
- **Color Jitter:** Simulates real-world variations in lighting conditions.
- **Gaussian Blur:** Adds robustness against noise and blurring artifacts.
- **Perspective Transform:** Models geometric distortions for diverse scenarios.
<img width="807" alt="image" src="https://github.com/user-attachments/assets/e611ae62-953b-4651-b739-b3e703bbaf10">

### Frequency-Domain Loss
Incorporates Fourier transform analysis to emphasize high-frequency details, crucial for sharp edges and textures.

The combined effect of these augmentations improves the model’s ability to generalize across diverse real-world conditions.

## Datasets

This project leverages a variety of datasets for training and evaluation:
1. **GoPro Dataset:** Synthetic motion blur images (1280x720 resolution).
   - Dataset: [Papers With Code](https://paperswithcode.com/dataset/gopro)
   - <img width="402" alt="image" src="https://github.com/user-attachments/assets/91e006bc-4e9a-4b75-b8c4-41de48c72782">
2. **RealBlur Dataset:** Real-world motion blur images with ground truth references.
   - Dataset: [RealBlur Dataset](https://cg.postech.ac.kr/research/realblur/)
   - Variants: RealBlur-R (RAW) and RealBlur-J (JPEG).
   - <img width="402" alt="image" src="https://github.com/user-attachments/assets/cee8c05d-6421-467f-a641-d6f7f07c8222">
3. **Ultra-High-Definition Motion Blurred (UHDM) Dataset:** High-resolution images (4K-6K) with complex blur patterns.
   - Dataset: [UHDM Dataset](https://github.com/HDCVLab/MC-Blur-Dataset)
   - <img width="402" alt="image" src="https://github.com/user-attachments/assets/517771b3-2283-4f6a-991d-5e8724a0fd6c">

## Evaluation Metrics
Performance is measured using:
- **PSNR (Peak Signal-to-Noise Ratio):** Quantifies image restoration quality.
- **SSIM (Structural Similarity Index):** Evaluates perceptual and structural fidelity.
- **DeltaE (Color Difference):** Measures color accuracy using the DeltaE2000 metric.
- **LPIPS (Learned Perceptual Image Patch Similarity):** Assesses perceptual similarity between restored and ground truth images.

## Results
- Achieved good performance on RealBlur-R and RealBlur-J datasets.
- Demonstrated strong generalization to the UHDM dataset, despite its challenging high-resolution scenarios.
- Significant improvements in robustness, as shown by hard positive and negative case analysis.
<img width="807" alt="image" src="https://github.com/user-attachments/assets/74651596-d0d6-48ce-ae2c-df91796afd42">
<img width="807" alt="image" src="https://github.com/user-attachments/assets/a564f129-5c5f-4ba3-9dc3-39ad272a2162">
<img width="807" alt="image" src="https://github.com/user-attachments/assets/5405569f-a066-4217-b9f4-cb176ad1f24b">

Some Examples:

<img width="807" alt="image" src="https://github.com/user-attachments/assets/52583876-fe51-49e4-a533-35cf92d63c6b">


<img width="807" alt="image" src="https://github.com/user-attachments/assets/130451c9-ede3-460c-a835-95b54843b08a">


## Installation

Please follow the steps below:

```bash
# Clone the repository
git clone https://github.com/hamzafer/image-deblurring
cd image-deblurring
```

Refer to the original [Restormer](https://github.com/swz30/Restormer/blob/main/INSTALL.md) repository for detailed setup instructions and dependencies.


## Usage
The model weights are available upon request.

~~Model weights can be found here: [Model Weights](https://studntnu-my.sharepoint.com/:f:/g/personal/muhamhz_ntnu_no/EvkFh21u5hZIr0TNXuW1HI0Bqyced3ZYhG_rxnOzyQD-Jw?e=vU817A)~~

### Running Inference
To test the improved model on your own images:

```bash
python demo.py --task Motion_Deblurring --input_dir /path/to/images --result_dir /path/to/save_results
```

### Training
Follow the instructions in the `train` directory to train the model on your dataset.

### Fine-Tuning
Fine-tuning scripts for RealBlur and UHDM datasets are available in the `fine_tune` directory.

## Acknowledgments
This work builds upon the [Restormer](https://github.com/swz30/Restormer) architecture by Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. We acknowledge their contributions and innovative work in developing an efficient transformer model for high-resolution image restoration.

### Citation
If you use this work or the Restormer architecture, please cite:

```bibtex
@article{akmaral2025efficient,
    title={Efficient Transformer for High Resolution Image Motion Deblurring},
    author={Akmaral, Amanturdieva and Zafar, Muhammad Hamza},
    journal={arXiv preprint arXiv:2501.18403},
    year={2025}
}

@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}

```
