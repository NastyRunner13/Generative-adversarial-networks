# SRGAN (Super-Resolution GAN)

This directory contains code for **SRGAN**, which aims to produce photo-realistic high-resolution images from low-resolution inputs.

## Architecture
![alt text](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png)

## Overview
SRGAN is composed of:
- **Generator**: A deep convolutional network that takes a low-resolution (LR) image and outputs a super-resolved (HR) image. It uses residual blocks and upsampling layers (sub-pixel convolutions) to gradually increase resolution.
- **Discriminator**: A convolutional network that distinguishes between generated (super-resolved) images and real high-resolution images.

The training includes:
- A perceptual loss that uses a pretrained VGG network to ensure the generated image is perceptually similar to the ground truth HR image.
- Adversarial loss to encourage the generator to produce images that look more like natural images.

## Contents
- **`config.py`**: Hyperparameters including image sizes, scaling factors, and paths to LR/HR datasets.
- **`dataset.py`**: Loads paired LR and HR images.
- **`discriminator.py`**: A CNN-based discriminator distinguishing generated HR from real HR.
- **`generator.py`**: A CNN-based generator with residual blocks and pixel-shuffle layers for upsampling.
- **`train.py`**: Training loop implementing SRGAN with perceptual and adversarial losses.
- **`utils.py`**: Utility functions for saving super-resolved images and loading checkpoints.

## Usage
```bash
cd srgan
python train.py
