import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


# Define Encoder
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

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.decoder(z)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
    
    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {avg_loss:.4f}")

    print("Training complete!")


def visualize_reconstruction(model, data_loader, artifacts_dir, num_images=5):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        reconstructed_images = model(images)

    images = images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original Images")
    axes[1, 0].set_title("Reconstructed Images")
    plt.savefig(artifacts_dir / "reconstruction_results.png")
    print("Plot saved to reconstruction_results.png")


def evaluate_reconstruction(model, data_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_mae = 0
    num_samples = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            mae = torch.abs(outputs - images).sum().item()
            total_mae += mae
            num_samples += images.numel()
    
    return total_mae / num_samples


def main(artifacts_dir):    
    transform = transforms.ToTensor()
    
    # Create train/val split
    full_dataset = datasets.MNIST(root="/datasets/cv_datasets/data", train=True, transform=transform, download=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load test set
    test_dataset = datasets.MNIST(root="/datasets/cv_datasets/data", train=False, transform=transform, download=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train model
    autoencoder = Autoencoder()
    train_autoencoder(autoencoder, train_loader, num_epochs=10)

    # Visualize
    visualize_reconstruction(autoencoder, train_loader, artifacts_dir)

    # Save encoder and decoder
    torch.save(autoencoder.encoder.state_dict(), artifacts_dir / "encoder_mnist.pth")
    torch.save(autoencoder.decoder.state_dict(), artifacts_dir / "decoder_mnist.pth")

    # Evaluate on all three sets
    train_mae = evaluate_reconstruction(autoencoder, train_loader)
    val_mae = evaluate_reconstruction(autoencoder, val_loader)
    test_mae = evaluate_reconstruction(autoencoder, test_loader)

    print(f"Training MAE: {train_mae:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

# #FOR VISUALIZATION FOR DRY:
# def main(artifacts_dir):    
#     import torchvision
#     from utils import plot_tsne
#     import shutil

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load encoder weights into the encoder submodule only
#     model_path = artifacts_dir / "encoder_mnist.pth"
#     model = Autoencoder().to(device)
#     model.encoder.load_state_dict(torch.load(model_path))  # ✅ FIXED
#     model.eval()

#     # Wrap encoder for t-SNE
#     class EncoderWrapper(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model

#         def forward(self, x):
#             return self.model.encode(x)

#     encoder = EncoderWrapper(model).to(device)

#     # Load test data
#     transform = transforms.ToTensor()
#     test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
#     test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

#     # Run and save t-SNE plots
#     plot_tsne(encoder, test_loader, device)
