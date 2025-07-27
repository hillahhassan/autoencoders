# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from utils import plot_tsne


# # Define Encoder
# class Encoder(nn.Module):
#     def __init__(self, latent_dim=128):
#         super(Encoder, self).__init__()
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, latent_dim)
#         )
#     def forward(self, x):
#         return self.conv_net(x)

# # Define NT-Xent loss function
# def nt_xent_loss(z, tau=0.5):
#     z = F.normalize(z, dim=1)
#     N = z.size(0) // 2
#     sim_matrix = torch.matmul(z, z.T) / tau
#     self_mask = torch.eye(2 * N, device=sim_matrix.device).bool()
#     sim_matrix = sim_matrix.masked_fill(self_mask, -1e9)
#     targets = torch.arange(0, 2*N, device=z.device)
#     targets = (targets + N) % (2 * N)
#     return F.cross_entropy(sim_matrix, targets)

# # Define Classifier
# class Classifier(nn.Module):
#     def __init__(self, latent_dim=128, num_classes=10):
#         super(Classifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)  # 10 classes for MNIST
#         )

#     def forward(self, x):
#         return self.fc(x)


# def main(artifacts_dir):

#     # Define data augmentation for contrastive learning
#     contrast_transforms = transforms.Compose([
#         transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.1,0.1))], p=0.8),
#         transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),  # New!
#         transforms.ToTensor()
#     ])


#     # Load MNIST dataset with ToTensor() applied
#     train_dataset = datasets.MNIST(root='/datasets/cv_datasets/data', train=True, download=True, transform=transforms.ToTensor())
#     val_dataset   = datasets.MNIST(root='/datasets/cv_datasets/data', train=False, download=True, transform=transforms.ToTensor())
#     train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)
#     val_loader    = DataLoader(val_dataset, batch_size=256, shuffle=False)


#     # Initialize model, optimizer
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     encoder = Encoder(latent_dim=128).to(device)
#     projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)).to(device)
#     optimizer = optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=5e-4)

#     # Training Loop
#     num_epochs = 10
#     tau = 0.1  # Instead of 0.5


#     for epoch in range(num_epochs):
#         encoder.train()
#         projection.train()
#         total_loss = 0.0

#         for images, _ in train_loader:
#             images = images.to(device)
#             imgs1 = torch.stack([contrast_transforms(transforms.ToPILImage()(img)) for img in images], dim=0).to(device)
#             imgs2 = torch.stack([contrast_transforms(transforms.ToPILImage()(img)) for img in images], dim=0).to(device)
#             batch = torch.cat([imgs1, imgs2], dim=0)

#             optimizer.zero_grad()
#             z = encoder(batch)
#             z = projection(z)
#             loss = nt_xent_loss(z, tau)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)

#         # Compute validation loss
#         encoder.eval()
#         projection.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for images, _ in val_loader:
#                 images = images.to(device)
#                 imgs1 = torch.stack([contrast_transforms(transforms.ToPILImage()(img)) for img in images], dim=0).to(device)
#                 imgs2 = torch.stack([contrast_transforms(transforms.ToPILImage()(img)) for img in images], dim=0).to(device)
#                 batch = torch.cat([imgs1, imgs2], dim=0)
#                 z = encoder(batch)
#                 z = projection(z)
#                 loss = nt_xent_loss(z, tau)
#                 val_loss += loss.item()
#         val_loss /= len(val_loader)

#         print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")


#     # In[2]:


#     encoder.eval()
#     all_features = []
#     all_labels = []

#     with torch.no_grad():
#         for images, labels in val_loader:  # Use validation set for visualization
#             images = images.to(device)
#             features = encoder(images)  # Encode images into latent space
#             all_features.append(features.cpu())  # Store on CPU for plotting
#             all_labels.append(labels.cpu())

#     all_features = torch.cat(all_features)
#     all_labels = torch.cat(all_labels)


#     # In[9]:


#     #sample fewer images
#     print("hiya")
#     import random

#     # Reduce dataset size for t-SNE (sample 2000 images instead of full set)
#     encoder.eval()
#     all_features = []
#     all_labels = []

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(device)
#             features = encoder(images)  # Encode images into latent space
#             all_features.append(features.cpu())  # Store on CPU for t-SNE
#             all_labels.append(labels.cpu())

#     all_features = torch.cat(all_features)
#     all_labels = torch.cat(all_labels)

#     # Select only 2000 random samples to speed up t-SNE
#     subset_indices = random.sample(range(len(all_features)), 2000)
#     all_features = all_features[subset_indices]
#     all_labels = all_labels[subset_indices]

#     # Now call plot_tsne()
#     plot_tsne(encoder, val_loader, device)
#     print("done")


#     # In[19]:

#     torch.save(encoder.state_dict(), artifacts_dir / "contrastive_encoder_mnist.pth")

#     # Freeze Encoder before Classifier Training
#     for param in encoder.parameters():
#         param.requires_grad = False

#     classifier = Classifier(latent_dim=128, num_classes=10).to(device)
#     classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()

#     # Train Classifier
#     num_epochs_classifier = 10
#     for epoch in range(num_epochs_classifier):
#         classifier.train()
#         total_loss = 0.0
#         correct = 0
#         total = 0

#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             with torch.no_grad():
#                 features = encoder(images).view(images.size(0), -1).to(device)
#             outputs = classifier(features.to(device))
#             loss = criterion(outputs, labels)

#             classifier_optimizer.zero_grad()
#             loss.backward()
#             classifier_optimizer.step()

#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)

#         train_loss = total_loss / len(train_loader)
#         train_acc = 100 * correct / total
#         print(f"Epoch {epoch+1}: Classifier Loss = {train_loss:.4f}, Accuracy = {train_acc:.2f}%")

#         # 🔥 **Evaluate Classifier on Validation Set after every epoch**
#         classifier.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0

#         with torch.no_grad():  # No gradient tracking during evaluation
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 features = encoder(images).view(images.size(0), -1)  # Extract features
#                 outputs = classifier(features.to(device))  # Pass through classifier
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 correct += predicted.eq(labels).sum().item()
#                 total += labels.size(0)

#         val_loss /= len(val_loader)
#         val_accuracy = 100 * correct / total
#         print(f"📌 Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
#         torch.save(classifier.state_dict(), artifacts_dir / "classifier_with_frozen_contrastive_encoder_mnist.pth")


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

