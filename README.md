# Semicon Hackathon 2026 — Image Restoration

Deep learning solution for restoring degraded semiconductor structure images: joint **denoising**, **deblurring**, and **2x super-resolution**, built for KLA's Semicon Hackathon 2026.

## Problem

Given a degraded grayscale image (noisy, blurred, and downsampled), reconstruct the original clean, full-resolution version.

The degradation pipeline `f: x → y` applied to ground truth images combines:
- **Speckle noise** — multiplicative grain, can push pixel values outside the original signal range
- **Gaussian blur** — softens edges and fine structure
- **Spatial downsampling** — 256×256 → 128×128 (2x reduction)

The model must learn the inverse mapping `f⁻¹: y → x`, and must do so **without over-smoothing** — the challenge explicitly penalizes solutions that remove noise by blurring away real detail.

**Test-time constraints:**
- Must generalize to out-of-distribution samples (semiconductor structures not seen during training)
- Inference speed is benchmarked — the model must be fast, not just accurate

## Dataset

| Split | Count | Contents |
|---|---|---|
| Train | 3,200 pairs | `GT/` (256×256, clean) + `NoisyLR/` (128×128, degraded) |
| Test | 400 images | `NoisyLR` only — in-distribution + out-of-distribution samples |

**Key data properties:**
- Grayscale, single channel
- GT pixel range: `[0, 1]`
- NoisyLR pixel range: exceeds `[0, 1]` (e.g. observed `[-0.08, 1.71]`) due to speckle noise — this is expected and must **not** be clipped or renormalized away

## Approach

### Data pipeline
- Custom PyTorch `Dataset` pairing `GT`/`NoisyLR` by filename
- No per-image normalization (preserves the true noisy signal range)
- Augmentation: horizontal/vertical flips + 90° rotations (safe transforms that don't interfere with the degradation signal itself)

### Model architecture
A compact residual CNN with pixel-shuffle upsampling:

- Head convolution → N residual blocks → global residual connection
- Pixel-shuffle (sub-pixel convolution) for artifact-free 2x learned upsampling
- Output clamped to `[0, 1]` (GT's known range)

Designed for a strong accuracy/speed tradeoff — avoids deep GAN/diffusion-style stacks in favor of a lightweight, fast architecture.

### Loss function
```
loss = L1(pred, target) + edge_weight * SobelEdgeLoss(pred, target)
```
- **L1**, not MSE — L1 produces sharper outputs; MSE biases toward blurry averages
- **Sobel gradient loss** — explicitly penalizes loss of edge/structural detail, directly targeting the "don't blur to denoise" requirement

### Training
- Adam optimizer, cosine annealing LR schedule
- Checkpointed every epoch (best-PSNR model + latest-epoch resume state) to survive Colab session disconnects
- 90/10 train/validation split

## Results

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Bicubic upsampling (baseline, no learning) | 22.29 | 0.5101 |
| **Trained model** | **27.98** | **0.7455** |

The model improves substantially over the naive baseline (+5.7 dB PSNR, SSIM nearly 50% higher), confirming genuine learned denoising and detail reconstruction rather than simple upsampling.

**Inference speed:** ~7 ms/image on GPU (T4), well within typical benchmarking budgets.

## Setup

```bash
pip install torch numpy scikit-image matplotlib
```

Designed to run in Google Colab with data mounted from Google Drive.

## Usage

1. Mount data and set paths to `GT/` and `NoisyLR/` folders
2. Run training:
   ```python
   python train.py
   ```
3. Checkpoints saved to `checkpoints/` (`best_model.pth`, `latest_checkpoint.pth`)
4. Run inference on test set:
   ```python
   python inference.py --input test/NoisyLR --output predictions/
   ```

## Repository Structure

```
.
├── Semicon_Hackathon.ipynb   # Main notebook: data pipeline, model, training, evaluation
├── checkpoints/              # Saved model weights
├── README.md
```

## Future Improvements

- Higher edge-loss weighting / SSIM-based loss term to further reduce residual over-smoothing in fine-texture regions
- Increased model capacity (more residual blocks / channels) given available inference speed headroom
- Additional augmentation strategies to strengthen out-of-distribution generalization

## Acknowledgements

Dataset and problem statement provided by KLA for Semicon Hackathon 2026.
