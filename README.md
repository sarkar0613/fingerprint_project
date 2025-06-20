# Fingerprint Recognition using Unsupervised Pre-training and Siamese Network Finetuning

This repository contains the implementation of my Master's thesis:
**"Research on Fingerprint Recognition Method Based on Unsupervised Pre-training and Siamese Network Architecture"**.

![image](https://github.com/user-attachments/assets/5ab58b22-2585-40ee-9942-ba46ac673eb1)


## 📁 Project Structure

```
src/
├── main.py                # Entry point for training
├── requirements.txt       # Required Python packages
├── data/                  # Dataloader and pair generation
├── models/                # Siamese architecture and model builder
├── train/                 # Loss functions and training loop
└── utils/                 # Utility and visualization tools
```

## 📂 Datasets
This project uses two fingerprint datasets at different stages of training:

1. PrintsGAN (Pretraining)
Description: A synthetic fingerprint dataset generated using Generative Adversarial Networks (GANs) developed by Michigan State University's PRIP Lab.
Purpose: Used for self-supervised pretraining via Barlow Twins to address the scarcity of labeled data.

Size:

35,000 unique identities

15 fingerprint images per identity

Total: 525,000 grayscale .tif images (256×256 resolution)

Naming Convention: FingerID_SampleID (e.g., 1_1.tif, 100_15.tif)

Features:

Simulates variations in lighting, deformation, and occlusion

Enables pretraining with large-scale, diverse data

🔬 2. Innolux Dataset (Fine-tuning and Evaluation)
Description: A proprietary fingerprint dataset collected using an optical, glass-based mobile fingerprint sensor under real-world conditions.

Purpose: Used to fine-tune the Siamese network and evaluate its performance under sensor variation and environmental challenges.

Structure:

30 subjects × 6 fingerprint classes = 180 unique fingerprints

Each fingerprint has:

20 enrollment images

50 verification images

Conditions Simulated:

ST – standard fluorescent lighting

100 – dry environment

90 – low-temperature environment (−5°C)

Challenges Addressed:

Partial prints

Incomplete ridge structures

Noise and environmental artifacts

Note: The Innolux dataset is proprietary and not publicly available. Please contact the authors for access if needed.

## 🚀 Features

- Self-supervised pre-training (Barlow Twins architecture)
- Siamese network for fingerprint verification
- Contrastive loss for fine-tuning
- Dataset preprocessing and dynamic pair generation

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/fingerprint_project.git
cd fingerprint_project/src

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Training

Prepare your `enroll` and `verify` dataset `.pt` files, then run:

```bash
bash train_single.sh
# or
bash train_ddp.sh
```

## 📊 Results

Results (e.g., accuracy, ROC curve) will be saved to the directory specified in `--result_dir`.

## 📚 License

This project is for academic use only. For commercial use, please contact the author.
