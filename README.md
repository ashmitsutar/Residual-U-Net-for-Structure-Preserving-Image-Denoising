# Residual-U-Net-for-Structure-Preserving-Image-Denoising
Residual U-Net for structure-preserving image denoising trained on 60K+ scientific FITS images. Implements grayscale-based training, curriculum learning (easy → hard → fine-tuning), and a hybrid L1 + SSIM + Gradient loss. Uses Y-channel denoising in YCbCr space during inference to preserve color fidelity while achieving high PSNR and SSIM.
