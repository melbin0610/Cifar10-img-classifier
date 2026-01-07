import torch
import torchvision.transforms as transforms
from PIL import Image

from cifar10_cnn import SimpleCNN   # uses the class in cifar10_cnn.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 classes [web:71]

# 1. Load the trained model weights
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.eval()

# 2. Use the SAME normalization as during training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

def predict_image(image_path: str):
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Apply transforms and add batch dimension
    tensor = transform(img).unsqueeze(0).to(device)   # shape (1, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)

    label = classes[predicted.item()]
    print(f"Predicted class: {label}")

if __name__ == "__main__":
    # Use your file name here
    predict_image("deer cifar.png")
