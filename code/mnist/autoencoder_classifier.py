import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Redefine Encoder (must match the one from 1.2.1)
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


def evaluate_accuracy(encoder, classifier, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    classifier.to(device)
    classifier.eval()
    encoder.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            latent = encoder(images)
            outputs = classifier(latent)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def train_frozen_encoder_classifier(encoder, classifier, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                latent = encoder(images)

            outputs = classifier(latent)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        val_accuracy = evaluate_accuracy(encoder, classifier, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

    print("Training complete!")


def visualize_many_predictions(encoder, classifier, test_loader, artifacts_dir, num_images=50, images_per_row=10):
    classifier.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)

    with torch.no_grad():
        latent = encoder(images)
        outputs = classifier(latent)
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
    plt.savefig(artifacts_dir / "many_predictions.png")
    print("Plot saved to many_predictions.png")


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

    pretrained_encoder = Encoder(latent_dim=128)
    pretrained_encoder.load_state_dict(torch.load(artifacts_dir / "encoder_mnist.pth"))
    pretrained_encoder.eval()
    for param in pretrained_encoder.parameters():
        param.requires_grad = False

    classifier = Classifier()
    train_frozen_encoder_classifier(pretrained_encoder, classifier, train_loader, val_loader, num_epochs=10)

    test_accuracy = evaluate_accuracy(pretrained_encoder, classifier, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    visualize_many_predictions(pretrained_encoder, classifier, test_loader, artifacts_dir)
    torch.save(classifier.state_dict(), artifacts_dir / "classifier_with_frozen_encoder_mnist.pth")

