# CIFAR-10 CNN Image Classifier

A PyTorch implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).[web:283][web:289]

---

## Features

- Trains a custom CNN on the CIFAR-10 dataset (32×32 color images, 10 classes, 50k train / 10k test).[web:283][web:289]  
- Saves the best model weights to `cifar10_cnn.pth` for later inference.  
- Includes a `predict_my_image.py` script to run predictions on your own images.  

---

## Project Structure

- `cifar10_cnn.py` – Defines the CNN architecture and training / evaluation loops.  
- `cifar10_cnn.pth` – Trained model weights for the CIFAR-10 classifier.  
- `predict_my_image.py` – Loads the trained model and predicts the class of a custom input image.  
- `data/` – (Optional) CIFAR-10 data directory; can be re-downloaded automatically by the training script.  
- `requirements.txt` – Python package dependencies.  

---

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/melbin0610/Cifar10-img-classifier.git
   cd Cifar10-img-classifier
2.Create and activate a virtual environment (recommended)

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux / macOS

3.Install dependencies:

pip install -r requirements.txt

