# CIFAR-10 Image Classifier 🎯

A Convolutional Neural Network (CNN) built with **PyTorch** to classify images from the **CIFAR-10** dataset into 10 object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.[web:110][web:112]

---

## Features

- Trains a custom CNN from scratch on the CIFAR-10 dataset  
- Achieves ~71% test accuracy after 5 epochs  
- Includes scripts/notebooks for:
  - Data loading and preprocessing
  - Model training and evaluation
  - Inference on custom images
- Saves and loads trained weights for reuse

---

## Dataset

This project uses the **CIFAR-10** dataset:[web:79][web:110][web:112]

- 60,000 color images, each of size **32×32×3 (RGB)**
- 50,000 training images and 10,000 test images
- 10 classes:
  - `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

The dataset is automatically downloaded via `torchvision.datasets.CIFAR10` the first time you run the training script.

---

## Project Structure

Adjust names if your files differ.

```text
cifar10-image-classifier/
├── data/                   # CIFAR-10 data (downloaded automatically)
├── models/
│   └── cifar10_cnn.pth     # Saved model weights (after training)
├── src/
│   ├── train.py            # Training loop
│   ├── model.py            # SimpleCNN architecture
│   └── utils.py            # Helper functions (metrics, plotting, etc.)
├── predict_my_image.py     # Inference on custom images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


## Installation

1. **Clone the repository**

```bash
git clone https://github.com/melbin0610/cifar10-image-classifier.git
cd cifar10-image-classifier

2.Create and activate a virtual environment 
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

3.Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt

4.Verify installation

python -c "import torch, torchvision; print('PyTorch version:', torch.__version__)"
