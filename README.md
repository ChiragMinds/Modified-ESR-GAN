**Ultrasound Image Enhancement Using Modified ESRGAN**
Overview

This repository implements a Modified Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) for ultrasound image enhancement.
The model improves the visual quality of medical ultrasound scans by reducing noise, restoring fine anatomical details, and enhancing structural clarity.
This work focuses on applying deep learning and GAN-based super-resolution techniques to grayscale medical imagery.

🔍 Motivation

Ultrasound images often suffer from low resolution and speckle noise, which obscure diagnostic details.
This project explores the use of GANs for enhancing ultrasound imagery to assist in computer-aided diagnosis and medical imaging research.

⚙️ Technical Highlights

Framework: PyTorch

Core Model: Modified ESRGAN with simplified RRDB architecture

Loss Functions: Custom combination of content, perceptual, and adversarial losses

Metrics: PSNR, SSIM, and MSE for image quality assessment

Device: CUDA / GPU-accelerated training

🧩 Features

Supports grayscale ultrasound image enhancement

Modular and well-structured PyTorch implementation

Includes training, validation, and inference pipelines

Compatible with custom medical image datasets

📈 Sample Results

The modified ESRGAN achieved significant improvement in quantitative and perceptual image quality compared to baseline ESRGAN models, demonstrating its capability for enhanced texture preservation and noise reduction in ultrasound scans.

🧰 Project Structure
Modified-ESR-GAN/
│
├── src/esrgan/              # Core modules
│   ├── data.py              # Dataset and dataloaders
│   ├── models.py            # Generator, Discriminator, VGG loss
│   ├── train.py             # Training pipeline
│   ├── test.py              # Inference and visualization
│   └── losses.py            # Evaluation metrics
│
├── notebooks/               # Experiments / Explorations
├── saved_models/            # Model checkpoints (ignored by Git)
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
└── README.md                # Project description

👨‍💻 Contributors

Chirag Chauhan
Himanshi Borad
Dhvani Maktuporia
Mayuri A. Mehta
Dheeraj Kumar Singh

📜 License

This project is licensed under the MIT License.
Usage or redistribution for academic purposes should include proper citation once the research is published.
