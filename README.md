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

1. **Vanilla GAN**
   - Based on the original GAN paper by Goodfellow et al.
   - Implements a simple architecture for learning from a dataset of images.

2. **Deep Convolutional GAN (DCGAN)**
   - Paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks."
   - Features a convolutional architecture for high-quality image generation.

3. **Conditional GAN (cGAN)**
   - Paper: "Conditional Generative Adversarial Nets."
   - Adds conditional inputs to guide the generation process.

4. **Wasserstein GAN (WGAN)**
   - Paper: "Wasserstein GAN."
   - Introduces the Wasserstein distance to improve GAN training stability.

5. **CycleGAN**
   - Paper: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks."
   - Focuses on image-to-image translation tasks.

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
1. Navigate to the specific GAN directory (e.g., `vanilla_gan/`):
   ```bash
   cd vanilla_gan
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

3. Model checkpoints and generated outputs will be saved in the `outputs/` directory.

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

1. **Generative Adversarial Networks** (Goodfellow et al.)
   [Paper](https://arxiv.org/abs/1406.2661)
2. **Unsupervised Representation Learning with Deep Convolutional GANs**
   [Paper](https://arxiv.org/abs/1511.06434)
3. **Conditional GANs**
   [Paper](https://arxiv.org/abs/1411.1784)
4. **Wasserstein GAN**
   [Paper](https://arxiv.org/abs/1701.07875)
5. **CycleGAN**
   [Paper](https://arxiv.org/abs/1703.10593)


Feel free to explore, experiment, and learn!
