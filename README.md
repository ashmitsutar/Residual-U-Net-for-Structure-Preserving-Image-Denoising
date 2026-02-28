📌 Overview

This repository contains an end-to-end implementation of a Residual U-Net based image denoising system designed for scientific astronomical imagery.

The model removes noise while preserving structural information and signal fidelity. Instead of predicting the clean image directly, the network learns the noise residual and subtracts it from the input.

📂 Dataset

The dataset was developed from publicly available JWST FITS observations.

~60,000 FITS images collected from official archives

Converted into normalized tensor representations

~30,000 noisy–clean paired samples generated

Curriculum splits:

Easy dataset (pretraining)

Hard dataset (robustness training)

Filter dataset (fine-tuning)

Training is performed on single-channel grayscale images to emphasize structural learning.

Dataset link: (Add Kaggle link here)

🧠 Model Architecture

Modified Residual U-Net with:

Encoder–decoder architecture

Skip connections for feature preservation

Dilated convolution bottleneck

Group Normalization for stable optimization

Residual learning formulation

Final reconstruction:

Output = Input − Predicted Noise

This improves convergence stability and preserves fine structures.

⚙️ Loss Function

Hybrid loss balancing accuracy and perceptual quality:

L1 Loss — pixel reconstruction

SSIM Loss — structural similarity

Gradient Loss — edge preservation

Final objective:

Loss = L1 + λ₁·SSIM + λ₂·GradientLoss
📈 Training Strategy
Stage 1 — Pretraining (Easy Dataset)

Learns global image statistics

Stabilizes early optimization

Stage 2 — Hard Dataset Training

Adapts to stronger noise distributions

Achieved:

PSNR ≈ 83.29 dB

SSIM ≈ 0.99998

Stage 3 — Fine-Tuning

Improves generalization across unseen samples

🔒 Stability Techniques

Gradient clipping

ReduceLROnPlateau scheduler

Early stopping

Validation-based checkpointing

🖼️ Inference Pipeline

Since training is grayscale-based:

Convert RGB → YCbCr

Apply denoising on Y (luminance) channel only

Preserve Cb and Cr channels

Merge channels to reconstruct final RGB image

This preserves color fidelity while removing structural noise.

📊 Evaluation Metrics

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

Both pixel accuracy and perceptual consistency are evaluated.

🗂️ Project Structure
├── dataset/
├── models/
│   └── unet.py
├── training/
│   └── train.py
├── inference/
│   └── predict.py
├── utils/
├── checkpoints/
└── README.md
🚀 Installation
git clone https://github.com/yourusername/repo-name.git
cd repo-name
pip install -r requirements.txt
🏃 Training
python train.py
🔍 Inference
python predict.py --image path_to_image
🧩 Technologies Used

Python

PyTorch

NumPy

Matplotlib

scikit-learn

pytorch-msssim

📜 License

Code released under MIT License.
Dataset follows CC-BY 4.0 attribution requirements.

👤 Author

Ashmit Sutar
B.Tech AI & ML | Machine Learning Enthusiast
