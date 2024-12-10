# Generative Adversarial Networks (GANs) with PyTorch

Welcome to the repository where I implement various types of Generative Adversarial Networks (GANs) inspired by research papers using PyTorch. This repository serves as a practical resource for learning and experimenting with state-of-the-art GAN architectures.

## Table of Contents
- [Overview](#overview)
- [Implemented GANs](#implemented-gans)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [References](#references)

---

## Overview
Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks, a Generator and a Discriminator, compete against each other to create synthetic data indistinguishable from real data. This repository implements GAN architectures from various research papers, focusing on:

- Reproducibility of results.
- Modular and clean PyTorch code.
- Simplified explanations and code comments.

The goal is to provide an educational platform for understanding and experimenting with GANs while exploring their diverse applications.

---

## Implemented GANs
Here are the GAN variants currently implemented in the repository:

1. **Deep Convolutional GAN (DCGAN)**
   - Paper: *"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"*
   - Features a convolutional architecture for high-quality image generation.

2. **Conditional GAN (cGAN)**
   - Paper: *"Conditional Generative Adversarial Nets"*
   - Adds conditional inputs to guide the generation process.

3. **Wasserstein GAN (WGAN)**
   - Paper: *"Wasserstein GAN"*
   - Introduces the Wasserstein distance to improve GAN training stability.

4. **CycleGAN**
   - Paper: *"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"*
   - Focuses on image-to-image translation tasks without paired training examples.

5. **SRGAN (Super-Resolution GAN)**
   - Paper: *"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"*
   - Designed for image super-resolution, combining perceptual and adversarial loss to generate high-quality HR images from LR inputs.

6. **ESRGAN (Enhanced SRGAN)**
   - Paper: *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"*
   - Builds upon SRGAN to further improve perceptual quality and fidelity through Residual-in-Residual Dense Blocks (RRDB).

More implementations will be added in the future. Check the repository regularly for updates!

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) (compatible with your GPU/CPU)
- Additional dependencies (listed in `requirements.txt`)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/NastyRunner13/Generative-adversarial-networks.git
   cd Generative-adversarial-networks
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training a GAN
1. Navigate to the specific GAN directory (e.g., `SRGAN/` or `ESRGAN/`):
   ```bash
   cd SRGAN
   ```
   
2. Run the training script:
   ```bash
   python train.py
   ```

3. Model checkpoints and generated outputs will be saved in the `outputs/` directory.

### Super-Resolution GANs (SRGAN & ESRGAN)
- **Data Preparation:**
  - For SRGAN and ESRGAN, prepare paired low-resolution (LR) and high-resolution (HR) images.
  - Place your training data in the `data/` directory or as specified in the `config.py` file for each GAN.

- **Training:**
  - Running `train.py` will load the dataset, train the Generator and Discriminator, and periodically save model checkpoints.
  
- **Evaluation and Testing:**
  - After training, use the provided test scripts or load the saved generator checkpoint to run inference on LR images and produce HR outputs.

### Experimenting with Hyperparameters
Modify the `config.py` file in each GAN folder to customize parameters such as learning rate, batch size, or model architecture.

### Visualizing Results
Generated outputs (e.g., images) are stored in the respective GAN folder under `outputs/`. Use any image viewer or notebook to visualize them.

---

## Contributing
Contributions are welcome! If you'd like to add a new GAN variant or improve the existing codebase, follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/new-gan
   ```
3. Commit your changes and push to your fork.
4. Submit a pull request describing your changes.

---

## References
This repository is inspired by the following research papers:

1. **Generative Adversarial Networks (Goodfellow et al.)**  
   [Paper](https://arxiv.org/abs/1406.2661)
   
2. **Unsupervised Representation Learning with DCGANs**  
   [Paper](https://arxiv.org/abs/1511.06434)
   
3. **Conditional GANs**  
   [Paper](https://arxiv.org/abs/1411.1784)
   
4. **Wasserstein GAN**  
   [Paper](https://arxiv.org/abs/1701.07875)
   
5. **CycleGAN**  
   [Paper](https://arxiv.org/abs/1703.10593)
   
6. **SRGAN**: Photo-Realistic Single Image Super-Resolution  
   [Paper](https://arxiv.org/abs/1609.04802)
   
7. **ESRGAN**: Enhanced Super-Resolution GANs  
   [Paper](https://arxiv.org/abs/1809.00219)

Feel free to explore, experiment, and learn!
