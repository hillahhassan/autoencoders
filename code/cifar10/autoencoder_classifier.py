# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.manifold import TSNE

# # Set random seed for reproducibility
# torch.manual_seed(42)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --------------------
# # (1) Define the Autoencoder
# # --------------------
# class CIFAR10Autoencoder(nn.Module):
#     def __init__(self, embedding_dim=128):
#         super(CIFAR10Autoencoder, self).__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             # Input: 3 x 32 x 32
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 32 x 16 x 16
            
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 8 x 8
            
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 128 x 4 x 4
            
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
        
#         # Embedding layer
#         self.flatten = nn.Flatten()
#         self.embedding = nn.Linear(256 * 4 * 4, embedding_dim)
        
#         # From embedding back to feature maps
#         self.unflatten = nn.Linear(embedding_dim, 256 * 4 * 4)
#         self.reshape = lambda x: x.view(-1, 256, 4, 4)
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             # Input: 256 x 4 x 4
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
#             nn.Tanh()  # Output in range [-1, 1]
#         )
    
#     def encode(self, x):
#         x = self.encoder(x)
#         x = self.flatten(x)
#         embedding = self.embedding(x)
#         return embedding
    
#     def decode(self, embedding):
#         x = self.unflatten(embedding)
#         x = self.reshape(x)
#         x = self.decoder(x)
#         return x
    
#     def forward(self, x):
#         embedding = self.encode(x)
#         reconstructed = self.decode(embedding)
#         return reconstructed, embedding

# # --------------------
# # (2) Data Loaders with Data Augmentation
# # --------------------
# def get_data_loaders(batch_size=128):
#     # For training, we apply random crop and horizontal flip
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), 
#                              (0.5, 0.5, 0.5))
#     ])
    
#     # For testing, just resize to tensor + normalize
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), 
#                              (0.5, 0.5, 0.5))
#     ])
    
#     # Load CIFAR-10 dataset
#     train_dataset = torchvision.datasets.CIFAR10(
#         root='/datasets/cv_datasets/data', 
#         train=True, 
#         download=True, 
#         transform=train_transform
#     )
#     test_dataset = torchvision.datasets.CIFAR10(
#         root='/datasets/cv_datasets/data', 
#         train=False, 
#         download=True, 
#         transform=test_transform
#     )
    
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=2
#     )
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=2
#     )
    
#     return train_loader, test_loader

# # --------------------
# # (3) Add Gaussian noise
# # --------------------
# def add_noise(inputs, noise_factor=0.2):
#     noise = torch.randn_like(inputs) * noise_factor
#     noisy_inputs = inputs + noise
#     return torch.clamp(noisy_inputs, -1., 1.)

# # --------------------
# # (4) Train the Autoencoder (with Denoising)
# # --------------------
# def train_autoencoder(model, train_loader, test_loader, artifacts_dir, num_epochs=50, learning_rate=0.001, noise_factor=0.2):
#     model = model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
#     train_losses = []
#     test_losses = []
    
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
        
#         for data in train_loader:
#             inputs, _ = data
#             inputs = inputs.to(device)
            
#             # Add noise
#             noisy_inputs = add_noise(inputs, noise_factor)
#             optimizer.zero_grad()
            
#             outputs, _ = model(noisy_inputs)
#             loss = criterion(outputs, inputs)  # Compare recon w/ clean
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         avg_train_loss = running_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
        
#         # Validation phase
#         model.eval()
#         test_loss = 0.0
        
#         with torch.no_grad():
#             for data in test_loader:
#                 inputs, _ = data
#                 inputs = inputs.to(device)
#                 noisy_inputs = add_noise(inputs, noise_factor)
                
#                 outputs, _ = model(noisy_inputs)
#                 loss = criterion(outputs, inputs)
#                 test_loss += loss.item()
        
#         avg_test_loss = test_loss / len(test_loader)
#         test_losses.append(avg_test_loss)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        
#         # Step the scheduler
#         scheduler.step(avg_test_loss)
    
#     # Plot the loss curves
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(test_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Denoising Autoencoder Training and Validation Loss')
#     plt.legend()
#     plt.savefig(artifacts_dir / 'denoising_autoencoder_loss.png')
#     plt.close()
    
#     return model

# # --------------------
# # (5) Visualize Reconstructed Images
# # --------------------
# def visualize_reconstruction(model, test_loader, artifacts_dir, noise_factor=0.2):
#     model.eval()
#     dataiter = iter(test_loader)
#     images, labels = next(dataiter)
#     images = images[:10].to(device)
    
#     # Create noisy versions
#     noisy_images = add_noise(images, noise_factor)
    
#     with torch.no_grad():
#         reconstructed, _ = model(noisy_images)
    
#     # Convert images from [-1, 1] to [0, 1] range
#     images = (images.cpu() + 1) / 2
#     noisy_images = (noisy_images.cpu() + 1) / 2
#     reconstructed = (reconstructed.cpu() + 1) / 2
    
#     plt.figure(figsize=(20, 6))
#     for i in range(10):
#         # Original
#         plt.subplot(3, 10, i + 1)
#         plt.imshow(images[i].permute(1, 2, 0))
#         plt.axis('off')
#         if i == 0:
#             plt.title('Original Images')
        
#         # Noisy
#         plt.subplot(3, 10, i + 11)
#         plt.imshow(noisy_images[i].permute(1, 2, 0))
#         plt.axis('off')
#         if i == 0:
#             plt.title('Noisy Images')
        
#         # Reconstructed
#         plt.subplot(3, 10, i + 21)
#         plt.imshow(reconstructed[i].permute(1, 2, 0))
#         plt.axis('off')
#         if i == 0:
#             plt.title('Reconstructed Images')
    
#     plt.savefig(artifacts_dir / 'denoising_reconstructions.png')
#     plt.close()

# # --------------------
# # (6) Visualize t-SNE
# # --------------------
# def visualize_embeddings_tsne(model, test_loader, artifacts_dir):
#     model.eval()
#     all_embeddings = []
#     all_labels = []
    
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             images = images.to(device)
#             _, embeddings = model(images)
#             all_embeddings.append(embeddings.cpu())
#             all_labels.append(labels)
    
#     all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
#     all_labels = torch.cat(all_labels, dim=0).numpy()
    
#     # Pick 2000 random samples for t-SNE
#     idx = np.random.choice(len(all_embeddings), 2000, replace=False)
#     sample_embeddings = all_embeddings[idx]
#     sample_labels = all_labels[idx]
    
#     print("Computing t-SNE projection...")
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
#     embeddings_tsne = tsne.fit_transform(sample_embeddings)
    
#     plt.figure(figsize=(10, 8))
#     classes = ['airplane', 'automobile', 'bird', 'cat', 
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    
#     for i, label in enumerate(np.unique(sample_labels)):
#         indices = sample_labels == label
#         plt.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], 
#                     color=colors[i], label=classes[i], alpha=0.7, s=20)
    
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.title('t-SNE visualization of CIFAR-10 embeddings (Denoising Autoencoder)')
#     plt.tight_layout()
#     plt.savefig(artifacts_dir / 'tsne_embeddings_denoising.png')
#     plt.close()

# # --------------------
# # (7) Classifier Head (Larger)
# # --------------------
# class ClassifierHead(nn.Module):
#     def __init__(self, embedding_dim=128, num_classes=10):
#         super(ClassifierHead, self).__init__()
#         # A bigger classifier head:
#         self.classifier = nn.Sequential(
#             nn.Linear(embedding_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
        
#     def forward(self, x):
#         return self.classifier(x)

# # --------------------
# # (8) Train Classifier on Frozen Encoder
# # --------------------

# def train_classifier(encoder, train_loader, test_loader, artifacts_dir, num_epochs=100, learning_rate=0.001):
#     for param in encoder.parameters():
#         param.requires_grad = False

#     classifier = ClassifierHead().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

#     train_accs = []
#     test_accs = []
#     best_test_acc = 0.0

#     for epoch in range(num_epochs):
#         classifier.train()
#         correct = 0
#         total = 0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             with torch.no_grad():
#                 embeddings = encoder.encode(inputs)

#             optimizer.zero_grad()
#             outputs = classifier(embeddings)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         train_acc = 100 * correct / total
#         train_accs.append(train_acc)

#         classifier.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 embeddings = encoder.encode(inputs)
#                 outputs = classifier(embeddings)

#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         test_acc = 100 * correct / total
#         test_accs.append(test_acc)

#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
#         scheduler.step(test_acc)

#         # Save the best model only
#         if test_acc > best_test_acc:
#             best_test_acc = test_acc
#             print(f"✅ New best test accuracy: {best_test_acc:.2f}% — saving model")
#             torch.save(classifier.state_dict(), artifacts_dir / 'cifar10_classifier_best.pth')

#     # Plot accuracy curves
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_accs, label='Training Accuracy')
#     plt.plot(test_accs, label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Classifier Training and Validation Accuracy')
#     plt.legend()
#     plt.savefig(artifacts_dir / 'classifier_accuracy.png')
#     plt.close()

#     return classifier


# # --------------------
# # (9) Main Execution
# # --------------------
# def main(artifacts_dir):
#     # Get data loaders (with augmentation)
#     train_loader, test_loader = get_data_loaders()
    
#     # Initialize the autoencoder
#     model = CIFAR10Autoencoder(embedding_dim=128)
    
#     # #4: Train longer: e.g. 100 epochs
#     # #5: Use a smaller noise factor: e.g. 0.1
#     print("Training denoising autoencoder...")
#     model = train_autoencoder(model, train_loader, test_loader, artifacts_dir, 
#                               num_epochs=150, 
#                               noise_factor=0.1)
    
#     # Save the trained model
#     torch.save(model.state_dict(), artifacts_dir / 'cifar10_denoising_autoencoder.pth')

#     # model = CIFAR10Autoencoder(embedding_dim=128)
#     # model.load_state_dict(torch.load('cifar10_denoising_autoencoder.pth'))
#     # model = model.to(device)

    
#     # Visualize reconstructions
#     print("Visualizing reconstructions...")
#     visualize_reconstruction(model, test_loader, artifacts_dir, noise_factor=0.1)
    
#     # Visualize embeddings with t-SNE
#     print("Visualizing t-SNE...")
#     visualize_embeddings_tsne(model, test_loader, artifacts_dir)
    
#     #6: Train a bigger classifier head
#     print("Training classifier on frozen encoder...")
#     classifier = train_classifier(model, train_loader, test_loader, artifacts_dir,
#                                   num_epochs=100,
#                                   learning_rate=0.001)
    
    
#     print("Done!")



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import random_split

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# (1) Define the Autoencoder
# --------------------
class CIFAR10Autoencoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CIFAR10Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 x 16 x 16
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 8 x 8
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 x 4 x 4
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Embedding layer
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(256 * 4 * 4, embedding_dim)
        
        # From embedding back to feature maps
        self.unflatten = nn.Linear(embedding_dim, 256 * 4 * 4)
        self.reshape = lambda x: x.view(-1, 256, 4, 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        embedding = self.embedding(x)
        return embedding
    
    def decode(self, embedding):
        x = self.unflatten(embedding)
        x = self.reshape(x)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        embedding = self.encode(x)
        reconstructed = self.decode(embedding)
        return reconstructed, embedding

# --------------------
# (2) Data Loaders with Data Augmentation
# --------------------
def get_data_loaders(batch_size=128):
    # For training, we apply random crop and horizontal flip
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                             (0.5, 0.5, 0.5))
    ])
    
    # For testing, just resize to tensor + normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), 
                             (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    # train_dataset = torchvision.datasets.CIFAR10(
    #     root='/datasets/cv_datasets/data', 
    #     train=True, 
    #     download=True, 
    #     transform=train_transform
    # )
    full_train_dataset = torchvision.datasets.CIFAR10(
    root='/datasets/cv_datasets/data', 
    train=True, 
    download=True, 
    transform=train_transform
    )
    
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    
    test_dataset = torchvision.datasets.CIFAR10(
        root='/datasets/cv_datasets/data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

# --------------------
# (3) Add Gaussian noise
# --------------------
def add_noise(inputs, noise_factor=0.2):
    noise = torch.randn_like(inputs) * noise_factor
    noisy_inputs = inputs + noise
    return torch.clamp(noisy_inputs, -1., 1.)

# --------------------
# (4) Train the Autoencoder (with Denoising)
# --------------------
# def train_autoencoder(model, train_loader, test_loader, artifacts_dir, num_epochs=50, learning_rate=0.001, noise_factor=0.2):
#     model = model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
#     train_losses = []
#     test_losses = []
    
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
        
#         for data in train_loader:
#             inputs, _ = data
#             inputs = inputs.to(device)
            
#             # Add noise
#             noisy_inputs = add_noise(inputs, noise_factor)
#             optimizer.zero_grad()
            
#             outputs, _ = model(noisy_inputs)
#             loss = criterion(outputs, inputs)  # Compare recon w/ clean
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         avg_train_loss = running_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
        
#         # Validation phase
#         model.eval()
#         test_loss = 0.0
        
#         with torch.no_grad():
#             for data in test_loader:
#                 inputs, _ = data
#                 inputs = inputs.to(device)
#                 noisy_inputs = add_noise(inputs, noise_factor)
                
#                 outputs, _ = model(noisy_inputs)
#                 loss = criterion(outputs, inputs)
#                 test_loss += loss.item()
        
#         avg_test_loss = test_loss / len(test_loader)
#         test_losses.append(avg_test_loss)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
        
#         # Step the scheduler
#         scheduler.step(avg_test_loss)
    
#     # Plot the loss curves
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(test_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Denoising Autoencoder Training and Validation Loss')
#     plt.legend()
#     plt.savefig(artifacts_dir / 'denoising_autoencoder_loss.png')
#     plt.close()
    
#     return model

def train_autoencoder(model, train_loader, val_loader, test_loader, artifacts_dir, num_epochs=50, learning_rate=0.001, noise_factor=0.2):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            noisy_inputs = add_noise(inputs, noise_factor)
            
            optimizer.zero_grad()
            outputs, _ = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, _ = data
                inputs = inputs.to(device)
                noisy_inputs = add_noise(inputs, noise_factor)
                outputs, _ = model(noisy_inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)
    
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Denoising Autoencoder Training and Validation Loss')
    plt.legend()
    plt.savefig(artifacts_dir / 'denoising_autoencoder_loss.png')
    plt.close()

    # Compute MAE
    def compute_mae(loader, label):
        model.eval()
        total_mae = 0
        total_pixels = 0
        with torch.no_grad():
            for data in loader:
                inputs, _ = data
                inputs = inputs.to(device)
                noisy_inputs = add_noise(inputs, noise_factor)
                outputs, _ = model(noisy_inputs)
                total_mae += torch.abs(outputs - inputs).sum().item()
                total_pixels += inputs.numel()
        mae = total_mae / total_pixels
        print(f'{label} MAE: {mae:.6f}')
        return mae

    compute_mae(train_loader, "Training")
    compute_mae(val_loader, "Validation")
    compute_mae(test_loader, "Test")

    return model


# --------------------
# (5) Visualize Reconstructed Images
# --------------------
def visualize_reconstruction(model, test_loader, artifacts_dir, noise_factor=0.2):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:10].to(device)
    
    # Create noisy versions
    noisy_images = add_noise(images, noise_factor)
    
    with torch.no_grad():
        reconstructed, _ = model(noisy_images)
    
    # Convert images from [-1, 1] to [0, 1] range
    images = (images.cpu() + 1) / 2
    noisy_images = (noisy_images.cpu() + 1) / 2
    reconstructed = (reconstructed.cpu() + 1) / 2
    
    plt.figure(figsize=(20, 6))
    for i in range(10):
        # Original
        plt.subplot(3, 10, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
        if i == 0:
            plt.title('Original Images')
        
        # Noisy
        plt.subplot(3, 10, i + 11)
        plt.imshow(noisy_images[i].permute(1, 2, 0))
        plt.axis('off')
        if i == 0:
            plt.title('Noisy Images')
        
        # Reconstructed
        plt.subplot(3, 10, i + 21)
        plt.imshow(reconstructed[i].permute(1, 2, 0))
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed Images')
    
    plt.savefig(artifacts_dir / 'denoising_reconstructions.png')
    plt.close()

# --------------------
# (6) Visualize t-SNE
# --------------------
def visualize_embeddings_tsne(model, test_loader, artifacts_dir):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            _, embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Pick 2000 random samples for t-SNE
    idx = np.random.choice(len(all_embeddings), 2000, replace=False)
    sample_embeddings = all_embeddings[idx]
    sample_labels = all_labels[idx]
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_tsne = tsne.fit_transform(sample_embeddings)
    
    plt.figure(figsize=(10, 8))
    classes = ['airplane', 'automobile', 'bird', 'cat', 
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    
    for i, label in enumerate(np.unique(sample_labels)):
        indices = sample_labels == label
        plt.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], 
                    color=colors[i], label=classes[i], alpha=0.7, s=20)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE visualization of CIFAR-10 embeddings (Denoising Autoencoder)')
    plt.tight_layout()
    plt.savefig(artifacts_dir / 'tsne_embeddings_denoising.png')
    plt.close()

# --------------------
# (7) Classifier Head (Larger)
# --------------------
class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=10):
        super(ClassifierHead, self).__init__()
        # A bigger classifier head:
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

# --------------------
# (8) Train Classifier on Frozen Encoder
# --------------------

def train_classifier(encoder, train_loader, test_loader, artifacts_dir, num_epochs=100, learning_rate=0.001):
    for param in encoder.parameters():
        param.requires_grad = False

    classifier = ClassifierHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    train_accs = []
    test_accs = []
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        classifier.train()
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                embeddings = encoder.encode(inputs)

            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_accs.append(train_acc)

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                embeddings = encoder.encode(inputs)
                outputs = classifier(embeddings)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        test_accs.append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        scheduler.step(test_acc)

        # Save the best model only
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"✅ New best test accuracy: {best_test_acc:.2f}% — saving model")
            torch.save(classifier.state_dict(), artifacts_dir / 'cifar10_classifier_best.pth')

    # Plot accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Classifier Training and Validation Accuracy')
    plt.legend()
    plt.savefig(artifacts_dir / 'classifier_accuracy.png')
    plt.close()

    return classifier


# --------------------
# (9) Main Execution
# --------------------
def main(artifacts_dir):
    # Get data loaders (with augmentation)
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Initialize the autoencoder
    model = CIFAR10Autoencoder(embedding_dim=128)
    
    # #4: Train longer: e.g. 100 epochs
    # #5: Use a smaller noise factor: e.g. 0.1
    print("Training denoising autoencoder...")
    model = train_autoencoder(model, train_loader, val_loader, test_loader, artifacts_dir, 
                              num_epochs=150, 
                              noise_factor=0.1)
    
    # Save the trained model
    torch.save(model.state_dict(), artifacts_dir / 'cifar10_denoising_autoencoder.pth')

    # model = CIFAR10Autoencoder(embedding_dim=128)
    # model.load_state_dict(torch.load('cifar10_denoising_autoencoder.pth'))
    # model = model.to(device)

    
    # Visualize reconstructions
    print("Visualizing reconstructions...")
    visualize_reconstruction(model, test_loader, artifacts_dir, noise_factor=0.1)
    
    # Visualize embeddings with t-SNE
    print("Visualizing t-SNE...")
    visualize_embeddings_tsne(model, test_loader, artifacts_dir)
    
    #6: Train a bigger classifier head
    print("Training classifier on frozen encoder...")
    classifier = train_classifier(model, train_loader, test_loader, artifacts_dir,
                                  num_epochs=100,
                                  learning_rate=0.001)
    
    
    print("Done!")


# #LATENT SPACE VISUALIZATION USING TRAINED MODEL:
# def main(artifacts_dir):
#     train_loader, val_loader, test_loader = get_data_loaders()

#     model = CIFAR10Autoencoder(embedding_dim=128)
#     model.load_state_dict(torch.load(artifacts_dir / 'cifar10_denoising_autoencoder.pth'))
#     model = model.to(device)

#     print("Generating t-SNE plots using saved encoder...")

#     class WrappedEncoder(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model

#         def forward(self, x):
#             return self.model.encode(x)

#     wrapped_encoder = WrappedEncoder(model)
#     from utils import plot_tsne
#     plot_tsne(wrapped_encoder, test_loader, device)

