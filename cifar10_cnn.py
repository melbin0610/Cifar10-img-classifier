import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Transforms (CIFAR-10 standard)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# 3. Datasets and loaders
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

classes = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# 4. Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (3,32,32) -> (32,16,16)
        x = self.pool(F.relu(self.conv2(x)))   # (32,16,16) -> (64,8,8)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. Create model, loss, optimizer
net = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train(num_epochs=5):
    net.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Plot training loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("CIFAR-10 CNN Training Loss")
    plt.grid(True)
    plt.show()

def test():
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test accuracy: {acc:.2f}%")
    return acc

def show_predictions():
    net.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    # unnormalize and move to CPU
    images = images.cpu() * 0.5 + 0.5
    images = images[:8]
    predicted = predicted[:8]

    plt.figure(figsize=(10, 4))
    for idx in range(8):
        img = images[idx].permute(1, 2, 0).numpy()
        plt.subplot(2, 4, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(classes[predicted[idx]])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train(num_epochs=5)
    acc = test()

    # Save trained weights for predict_my_image.py
    torch.save(net.state_dict(), "cifar10_cnn.pth")
    print("Saved model to cifar10_cnn.pth")

    # Optional: show some test predictions
    show_predictions()
