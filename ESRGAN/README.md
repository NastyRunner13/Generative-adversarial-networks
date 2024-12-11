# ESRGAN (Enhanced Super-Resolution GAN)

This directory contains code for **ESRGAN**, an improved version of SRGAN that further enhances perceptual quality.

## Architecture
![alt text](https://user-images.githubusercontent.com/41912303/121768709-d2914700-cb5f-11eb-90f5-c37b86476bb8.png)

## Overview
ESRGAN improves upon SRGAN through:
- **Generator**: Uses Residual-in-Residual Dense Blocks (RRDB) instead of standard residual blocks, removing batch normalization layers and improving the network’s ability to learn more complex features.
- **Discriminator**: Similar to SRGAN’s discriminator but tuned for ESRGAN’s higher fidelity results.
- **Losses**: Combines adversarial loss, perceptual loss (using VGG features), and pixel-wise loss. The adversarial loss uses a relativistic discriminator to compare real and fake images.

This architecture typically produces sharper and more realistic details in the upscaled images.

## Contents
- **`config.py`**: Hyperparameters (learning rate, dataset paths, scale factor).
- **`dataset.py`**: Loads LR and HR image pairs, applies necessary transformations.
- **`discriminator.py`**: A CNN-based relativistic discriminator.
- **`generator.py`**: A generator network built with RRDB blocks for improved super-resolution quality.
- **`train.py`**: Training loop that incorporates adversarial, perceptual, and pixel losses to produce high-quality super-resolved images.
- **`utils.py`**: Utilities for saving enhanced images, model checkpoints, and logging metrics.

## Usage
```bash
cd esrgan
python train.py
