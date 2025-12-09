<h1 align="center">🧠 Ultrasound Image Enhancement Using Modified ESRGAN</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red.svg" alt="Framework">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <a href="https://ieeexplore.ieee.org/document/11233129">
    <img src="https://img.shields.io/badge/Published%20on-IEEE%20Xplore-blue.svg" alt="IEEE Publication">
  </a>
</p>

---

## 🧾 Overview
This repository implements a **Modified Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)** for **ultrasound image enhancement**.  
The model improves the visual quality of medical ultrasound scans by **reducing noise**, **restoring fine anatomical details**, and **enhancing structural clarity**.  
This work explores **GAN-based super-resolution** for grayscale medical imagery.

---

## 🔍 Motivation
Ultrasound images often suffer from **low resolution and speckle noise**, which obscure diagnostic details.  
This project applies deep learning and GAN-based super-resolution to improve ultrasound image quality for **better computer-aided diagnosis and medical analysis**.

---

## ⚙️ Technical Highlights
- **Framework:** PyTorch  
- **Model:** Modified ESRGAN with simplified RRDB architecture  
- **Loss Functions:** Custom combination of content, perceptual, and adversarial losses  
- **Evaluation Metrics:** PSNR, SSIM, and MSE  
- **Hardware:** CUDA / GPU-accelerated training  

---

## 🧩 Features
- Supports **grayscale ultrasound image enhancement**  
- Modular and clean **PyTorch implementation**  
- Includes **training**, **validation**, and **inference** pipelines  
- Compatible with **custom medical datasets**  

---

## 📈 Sample Results
The modified ESRGAN achieved **significant improvement** in both quantitative and perceptual image quality compared to baseline ESRGAN models.  
It demonstrated improved **texture preservation**, **noise reduction**, and **structural similarity** across multiple ultrasound categories.  

---

## 🧰 Project Structure
```bash
my-esrgan-project/
│
├── src/
│   └── esrgan/
│       ├── data.py              # Dataset and dataloaders
│       ├── models.py            # Generator, Discriminator, VGG loss
│       ├── train.py             # Training pipeline
│       ├── test.py              # Inference and visualization
│       └── losses.py            # Evaluation metrics
│
├── notebooks/                   # Jupyter notebooks for experiments
├── saved_models/                # Model checkpoints (ignored by Git)
├── requirements.txt             # Dependencies list
├── LICENSE                      # MIT License file
└── README.md                    # Project documentation
```

## 📄 Publication

This research has been **officially published on IEEE Xplore**:

> **C. Chauhan, F. Patel, C. Tejani, and D. Kumar Singh**,  
> *"Ultrasound Image Enhancement Using Modified ESRGAN,"*  
> *2025 International Conference on Intelligent Systems and Image Processing (ICISIP), IEEE, 2025.*  
> [DOI: 10.1109/ICISIP.2025.11233129](https://ieeexplore.ieee.org/document/11233129)

🔗 **Direct Link:** [IEEE Xplore Publication Page →](https://ieeexplore.ieee.org/document/11233129)

If you use this repository or reproduce results, please cite the above publication.

---

## 📜 License

This project is licensed under the **MIT License**.

Use of this repository or its code for **academic or research purposes** should include proper citation of the above IEEE paper.

---

<p align="center">
  <b>© 2025 Chirag Chauhan — All Rights Reserved.</b><br>
  Published work available on <a href="https://ieeexplore.ieee.org/document/11233129">IEEE Xplore</a>.
</p>
