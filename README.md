# Fingerprint Recognition using Unsupervised Pre-training and Siamese Network

This repository contains the implementation of my Master's thesis:
**"Research on Fingerprint Recognition Method Based on Unsupervised Pre-training and Siamese Network Architecture"**.

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
