import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class ClassificationModel(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(ClassificationModel, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

def evaluate_accuracy(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def train_encoder_and_classifier(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        val_acc = evaluate_accuracy(model, val_loader)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    print("Training complete!")

def evaluate_joint_model(model, test_loader):
    acc = evaluate_accuracy(model, test_loader)
    print(f"Test Accuracy: {acc:.2f}%")

def visualize_joint_predictions(model, test_loader, artifacts_dir, num_images=50, images_per_row=10):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    predicted = predicted.cpu()
    labels = labels.cpu()

    rows = num_images // images_per_row
    fig, axes = plt.subplots(rows, images_per_row, figsize=(images_per_row * 2, rows * 2))

    for i in range(num_images):
        row = i // images_per_row
        col = i % images_per_row
        axes[row, col].imshow(images[i].squeeze(), cmap="gray")
        axes[row, col].set_title(f"P: {predicted[i].item()}\nT: {labels[i].item()}", fontsize=10)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(artifacts_dir / "joint_predictions.png")
    print("Plot saved to joint_predictions.png")

def main(artifacts_dir):
    transform = transforms.ToTensor()
    full_dataset = datasets.MNIST(root="/datasets/cv_datasets/data", train=True, transform=transform, download=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = datasets.MNIST(root="/datasets/cv_datasets/data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ClassificationModel()
    train_encoder_and_classifier(model, train_loader, val_loader, num_epochs=10)
    torch.save(model.state_dict(), artifacts_dir / "encoder_classifier_mnist.pth")

    evaluate_joint_model(model, test_loader)
    visualize_joint_predictions(model, test_loader, artifacts_dir)
    
#VISUALIZATIONS FOR DRY USING TRAINED MODEL
    
# def plot_tsne_main(artifacts_dir):
#     import torchvision
#     from utils import plot_tsne
#     import shutil

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load trained model
#     model_path = artifacts_dir / "encoder_classifier_mnist.pth"  # assuming you reused this filename
#     model = ClassificationModel().to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     # Wrap the encoder for t-SNE
#     class EncoderWrapper(nn.Module):
#         def __init__(self, encoder):
#             super().__init__()
#             self.encoder = encoder

#         def forward(self, x):
#             return self.encoder(x)

#     encoder = EncoderWrapper(model.encoder).to(device)

#     # Load test data
#     transform = transforms.ToTensor()
#     test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
#     test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

#     # Run t-SNE
#     plot_tsne(encoder, test_loader, device)

#     # Save plots to artifacts directory
#     shutil.copy("latent_tsne.png", artifacts_dir / "latent_tsne_contrastive.png")
#     shutil.copy("image_tsne.png", artifacts_dir / "image_tsne_contrastive.png")

