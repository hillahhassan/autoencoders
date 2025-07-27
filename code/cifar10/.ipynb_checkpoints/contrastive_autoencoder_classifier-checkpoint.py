# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# # Set random seed for reproducibility
# torch.manual_seed(42)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Autoencoder architecture (without skip connections)
# class CIFAR10Autoencoder(nn.Module):
#     def __init__(self, embedding_dim=128):  # Changed to 128-dim embedding
#         super(CIFAR10Autoencoder, self).__init__()
        
#         # Encoder
#         self.encoder_conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 x 16 x 16
        
#         self.encoder_conv2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 x 8 x 8
        
#         self.encoder_conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 x 4 x 4
        
#         self.encoder_conv4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
#         )
        
#         # Embedding layer
#         self.flatten = nn.Flatten()
#         self.embedding = nn.Sequential(
#             nn.Linear(512 * 4 * 4, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, embedding_dim)  # Final embedding dimension of 128
#         )
        
#         # Projection head for contrastive learning
#         self.projection = nn.Sequential(
#             nn.Linear(embedding_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )
        
#         # From embedding back to feature maps
#         self.unflatten = nn.Sequential(
#             nn.Linear(embedding_dim, 512),  # From 128-dim embedding
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 512 * 4 * 4)
#         )
#         self.reshape = lambda x: x.view(-1, 512, 4, 4)
        
#         # Decoder with transpose convolutions
#         self.decoder_conv1 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
        
#         self.decoder_conv2 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
        
#         self.decoder_conv3 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
        
#         self.decoder_conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
#         self.tanh = nn.Tanh()
    
#     def encode(self, x):
#         # Forward through encoder
#         x = self.encoder_conv1(x)
#         x = self.pool1(x)
#         x = self.encoder_conv2(x)
#         x = self.pool2(x)
#         x = self.encoder_conv3(x)
#         x = self.pool3(x)
#         x = self.encoder_conv4(x)
        
#         x = self.flatten(x)
#         embedding = self.embedding(x)
#         return embedding
    
#     def project(self, embedding):
#         return self.projection(embedding)
    
#     def decode(self, embedding):
#         x = self.unflatten(embedding)
#         x = self.reshape(x)
        
#         x = self.decoder_conv1(x)
#         x = self.decoder_conv2(x)
#         x = self.decoder_conv3(x)
#         x = self.decoder_conv4(x)
#         x = self.tanh(x)
#         return x
    
#     def forward(self, x):
#         embedding = self.encode(x)
#         projection = self.project(embedding)
#         reconstructed = self.decode(embedding)
#         return reconstructed, embedding, projection

# # Contrastive loss (InfoNCE/NT-Xent)
# class NTXentLoss(nn.Module):
#     def __init__(self, temperature=0.5):
#         super(NTXentLoss, self).__init__()
#         self.temperature = temperature
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, z_i, z_j):
#         z_i = nn.functional.normalize(z_i, dim=1)
#         z_j = nn.functional.normalize(z_j, dim=1)

#         batch_size = z_i.size(0)
#         representations = torch.cat([z_i, z_j], dim=0)  # (2N, D)

#         # Cosine similarity matrix
#         sim_matrix = torch.matmul(representations, representations.T) / self.temperature

#         # Mask self-similarities
#         mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim_matrix.device)
#         sim_matrix.masked_fill_(mask, -float('inf'))

#         # Positive pairs: i-th in z_i matches i-th in z_j
#         positives = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(sim_matrix.device)

#         labels = positives
#         loss = self.criterion(sim_matrix, labels)
#         return loss

# # Custom dataset for contrastive learning
# class ContrastiveCIFAR10(Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform
        
#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
        
#         if self.transform:
#             img1 = self.transform(img)
#             img2 = self.transform(img)
#         else:
#             img1, img2 = img, img
            
#         return img1, img2, label
    
#     def __len__(self):
#         return len(self.dataset)

# # Data loading with augmentation for contrastive learning
# def get_contrastive_data_loaders(batch_size=128):
#     # Base transformations
#     base_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     # Strong augmentation for contrastive views
#     contrastive_transform = transforms.Compose([
#         transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     # Load base datasets
#     # train_dataset_base = torchvision.datasets.CIFAR10(
#     #     root='/datasets/cv_datasets/data', train=True, download=True, transform=base_transform)
#     train_dataset_base = torchvision.datasets.CIFAR10(
#     root='/datasets/cv_datasets/data', train=True, download=True, transform=None)  # No transform here
#     test_dataset = torchvision.datasets.CIFAR10(
#         root='/datasets/cv_datasets/data', train=False, download=True, transform=base_transform)
    
#     # Create contrastive dataset for training
#     train_dataset = ContrastiveCIFAR10(train_dataset_base, contrastive_transform)
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
#     return train_loader, test_loader

# # Add noise to input for denoising training
# def add_noise(inputs, noise_factor=0.2):
#     noise = torch.randn_like(inputs) * noise_factor
#     noisy_inputs = inputs + noise
#     # Clip to ensure we stay in the valid range [-1, 1]
#     return torch.clamp(noisy_inputs, -1., 1.)

# # Training function with combined reconstruction and contrastive learning
# def train_contrastive_encoder(model, train_loader, test_loader, artifacts_dir, num_epochs=50, learning_rate=0.001, noise_factor=0.2, lambda_contrastive=0.5):
#     model = model.to(device)
    
#     rec_criterion = nn.MSELoss()
#     contrastive_criterion = NTXentLoss(temperature=0.1)
    
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     train_rec_losses = []
#     train_contrastive_losses = []
#     train_total_losses = []
#     test_losses = []

#     best_test_loss = float('inf')  # Track lowest validation loss
#     best_model_state = None        # To store best model weights
    
#     start_time = time.time()
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_rec_loss = 0.0
#         running_contrastive_loss = 0.0
#         running_total_loss = 0.0
        
#         for data in train_loader:
#             img1, img2, _ = data
#             img1, img2 = img1.to(device), img2.to(device)
            
#             noisy_img1 = add_noise(img1, noise_factor)
#             noisy_img2 = add_noise(img2, noise_factor)
            
#             optimizer.zero_grad()
            
#             outputs1, _, proj1 = model(noisy_img1)
#             outputs2, _, proj2 = model(noisy_img2)
            
#             rec_loss1 = rec_criterion(outputs1, img1)
#             rec_loss2 = rec_criterion(outputs2, img2)
#             rec_loss = (rec_loss1 + rec_loss2) / 2
            
#             contrastive_loss = contrastive_criterion(proj1, proj2)
#             total_loss = rec_loss + lambda_contrastive * contrastive_loss
            
#             total_loss.backward()
#             optimizer.step()
            
#             running_rec_loss += rec_loss.item()
#             running_contrastive_loss += contrastive_loss.item()
#             running_total_loss += total_loss.item()
        
#         avg_rec_loss = running_rec_loss / len(train_loader)
#         avg_contrastive_loss = running_contrastive_loss / len(train_loader)
#         avg_total_loss = running_total_loss / len(train_loader)
        
#         train_rec_losses.append(avg_rec_loss)
#         train_contrastive_losses.append(avg_contrastive_loss)
#         train_total_losses.append(avg_total_loss)
        
#         # Validation
#         model.eval()
#         test_loss = 0.0
        
#         with torch.no_grad():
#             for data in test_loader:
#                 inputs, _ = data
#                 inputs = inputs.to(device)
#                 noisy_inputs = add_noise(inputs, noise_factor)
                
#                 outputs, _, _ = model(noisy_inputs)
#                 loss = rec_criterion(outputs, inputs)
#                 test_loss += loss.item()
        
#         avg_test_loss = test_loss / len(test_loader)
#         test_losses.append(avg_test_loss)
        
#         # Save best model
#         if avg_test_loss < best_test_loss:
#             best_test_loss = avg_test_loss
#             best_model_state = model.state_dict()
#             torch.save(best_model_state, artifacts_dir / 'best_contrastive_encoder.pth')
        
#         scheduler.step()
        
#         elapsed_time = time.time() - start_time
#         print(f'Epoch [{epoch+1}/{num_epochs}], '
#               f'Rec Loss: {avg_rec_loss:.4f}, '
#               f'Contrastive Loss: {avg_contrastive_loss:.4f}, '
#               f'Total Loss: {avg_total_loss:.4f}, '
#               f'Test Loss: {avg_test_loss:.4f}, '
#               f'LR: {scheduler.get_last_lr()[0]:.6f}, '
#               f'Time: {elapsed_time:.1f}s')
    
#     # Plot training curves
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_rec_losses, label='Training Reconstruction Loss')
#     plt.plot(train_contrastive_losses, label='Training Contrastive Loss')
#     plt.plot(train_total_losses, label='Training Total Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Losses')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(test_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Validation Loss')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(artifacts_dir / 'contrastive_encoder_loss.png')
#     plt.close()
    
#     return model

# # Classifier head remains the same
# class ClassifierHead(nn.Module):
#     def __init__(self, embedding_dim=128, num_classes=10):
#         super(ClassifierHead, self).__init__()
        
#         self.classifier = nn.Sequential(
#             nn.Linear(embedding_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )
        
#     def forward(self, x):
#         return self.classifier(x)

# def linear_evaluation(encoder, train_loader, test_loader, artifacts_dir, num_epochs=50, learning_rate=0.001):
#     base_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     train_dataset = torchvision.datasets.CIFAR10(
#         root='/datasets/cv_datasets/data', train=True, download=True, transform=base_transform)
    
#     train_loader_linear = DataLoader(
#         train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    
#     encoder.eval()
#     for param in encoder.parameters():
#         param.requires_grad = False
    
#     classifier = ClassifierHead(embedding_dim=128).to(device)
#     criterion = nn.CrossEntropyLoss()
    
#     optimizer = optim.AdamW(
#         classifier.parameters(), 
#         lr=learning_rate,
#         weight_decay=1e-4
#     )
    
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=learning_rate,
#         steps_per_epoch=len(train_loader_linear),
#         epochs=num_epochs
#     )
    
#     train_accs = []
#     test_accs = []
#     best_acc = 0.0
#     best_model_state = None
    
#     # Precompute embeddings
#     train_embeddings = []
#     train_labels = []
#     test_embeddings = []
#     test_labels = []
    
#     print("Precomputing embeddings...")
#     with torch.no_grad():
#         for data in train_loader_linear:
#             inputs, labels = data
#             inputs = inputs.to(device)
#             embeddings = encoder.encode(inputs)
#             train_embeddings.append(embeddings.cpu())
#             train_labels.append(labels)
        
#         for data in test_loader:
#             inputs, labels = data
#             inputs = inputs.to(device)
#             embeddings = encoder.encode(inputs)
#             test_embeddings.append(embeddings.cpu())
#             test_labels.append(labels)
    
#     train_embeddings = torch.cat(train_embeddings, dim=0)
#     train_labels = torch.cat(train_labels, dim=0)
#     test_embeddings = torch.cat(test_embeddings, dim=0)
#     test_labels = torch.cat(test_labels, dim=0)
    
#     from torch.utils.data import TensorDataset
    
#     train_dataset = TensorDataset(train_embeddings, train_labels)
#     test_dataset = TensorDataset(test_embeddings, test_labels)
    
#     train_loader_emb = DataLoader(train_dataset, batch_size=256, shuffle=True)
#     test_loader_emb = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
#     for epoch in range(num_epochs):
#         classifier.train()
#         correct = 0
#         total = 0
        
#         for embeddings, labels in train_loader_emb:
#             embeddings, labels = embeddings.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = classifier(embeddings)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
            
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         train_acc = 100 * correct / total
#         train_accs.append(train_acc)
        
#         # Validation
#         classifier.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for embeddings, labels in test_loader_emb:
#                 embeddings, labels = embeddings.to(device), labels.to(device)
#                 outputs = classifier(embeddings)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         test_acc = 100 * correct / total
#         test_accs.append(test_acc)
        
#         # Save best model
#         if test_acc > best_acc:
#             best_acc = test_acc
#             best_model_state = classifier.state_dict()
#             torch.save(best_model_state, artifacts_dir / 'best_linear_classifier.pth')
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
#     # Plot the accuracy curves
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_accs, label='Training Accuracy')
#     plt.plot(test_accs, label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Linear Evaluation Accuracy')
#     plt.legend()
#     plt.savefig(artifacts_dir / 'linear_eval_accuracy.png')
#     plt.close()
    
#     print(f"Best test accuracy: {best_acc:.2f}%")
    
#     return classifier, best_acc


# # Visualization of embeddings with t-SNE
# def visualize_embeddings(encoder, test_loader, artifacts_dir, num_samples=1000):
#     encoder.eval()
#     embeddings = []
#     labels = []
    
#     with torch.no_grad():
#         for data in test_loader:
#             inputs, target = data
#             inputs = inputs.to(device)
#             emb = encoder.encode(inputs)
#             embeddings.append(emb.cpu().numpy())
#             labels.append(target.numpy())
            
#             if len(np.concatenate(embeddings)) >= num_samples:
#                 break
    
#     embeddings = np.concatenate(embeddings)[:num_samples]
#     labels = np.concatenate(labels)[:num_samples]
    
#     # t-SNE dimensionality reduction
#     from sklearn.manifold import TSNE
    
#     tsne = TSNE(n_components=2, random_state=42)
#     embeddings_2d = tsne.fit_transform(embeddings)
    
#     # Plot with colors based on class labels
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
#     plt.colorbar(scatter, ticks=range(10))
#     plt.title('t-SNE visualization of embeddings')
#     plt.savefig(artifacts_dir / 'tsne_embeddings.png')
#     plt.close()

# # Main execution
# def main(artifacts_dir):
#     print(f"Using device: {device}")
    
#     # Get data loaders for contrastive learning
#     train_loader, test_loader = get_contrastive_data_loaders(batch_size=128)
    
#     # Initialize the autoencoder with projection head
#     embedding_dim = 128
#     noise_factor = 0.15
#     lambda_contrastive = 1.0  # Weight for contrastive loss
    
#     model = CIFAR10Autoencoder(embedding_dim=embedding_dim)
#     print(f"Training contrastive encoder with noise factor {noise_factor}...")
#     model = train_contrastive_encoder(
#         model, 
#         train_loader, 
#         test_loader, 
#         artifacts_dir,
#         num_epochs=50,  
#         learning_rate=0.001,
#         noise_factor=noise_factor,
#         lambda_contrastive=lambda_contrastive
#     )
    
#     # Save the trained model
#     torch.save(model.state_dict(), artifacts_dir / 'cifar10_contrastive_encoder.pth')
    
#     # Visualize the learned embeddings
#     print("Visualizing embeddings with t-SNE...")
#     visualize_embeddings(model, test_loader, artifacts_dir)
    
#     # Train classifier with linear evaluation protocol
#     print("Performing linear evaluation on learned representations...")
#     classifier, best_acc = linear_evaluation(
#         model, 
#         train_loader, 
#         test_loader,
#         artifacts_dir,
#         num_epochs=50,
#         learning_rate=0.003
#     )
    
#     print(f"Final best test accuracy: {best_acc:.2f}%")
#     print("Done!")


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import time

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder architecture (without skip connections)
class CIFAR10Autoencoder(nn.Module):
    def __init__(self, embedding_dim=128):  # Changed to 128-dim embedding
        super(CIFAR10Autoencoder, self).__init__()
        
        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 x 16 x 16
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128 x 8 x 8
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 x 4 x 4
        
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Embedding layer
        self.flatten = nn.Flatten()
        self.embedding = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)  # Final embedding dimension of 128
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # From embedding back to feature maps
        self.unflatten = nn.Sequential(
            nn.Linear(embedding_dim, 512),  # From 128-dim embedding
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512 * 4 * 4)
        )
        self.reshape = lambda x: x.view(-1, 512, 4, 4)
        
        # Decoder with transpose convolutions
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
    
    def encode(self, x):
        # Forward through encoder
        x = self.encoder_conv1(x)
        x = self.pool1(x)
        x = self.encoder_conv2(x)
        x = self.pool2(x)
        x = self.encoder_conv3(x)
        x = self.pool3(x)
        x = self.encoder_conv4(x)
        
        x = self.flatten(x)
        embedding = self.embedding(x)
        return embedding
    
    def project(self, embedding):
        return self.projection(embedding)
    
    def decode(self, embedding):
        x = self.unflatten(embedding)
        x = self.reshape(x)
        
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_conv3(x)
        x = self.decoder_conv4(x)
        x = self.tanh(x)
        return x
    
    def forward(self, x):
        embedding = self.encode(x)
        projection = self.project(embedding)
        reconstructed = self.decode(embedding)
        return reconstructed, embedding, projection

# Contrastive loss (InfoNCE/NT-Xent)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        batch_size = z_i.size(0)
        representations = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        # Mask self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix.masked_fill_(mask, -float('inf'))

        # Positive pairs: i-th in z_i matches i-th in z_j
        positives = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(sim_matrix.device)

        labels = positives
        loss = self.criterion(sim_matrix, labels)
        return loss

# Custom dataset for contrastive learning
class ContrastiveCIFAR10(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1, img2 = img, img
            
        return img1, img2, label
    
    def __len__(self):
        return len(self.dataset)

# Data loading with augmentation for contrastive learning
def get_contrastive_data_loaders(batch_size=128):
    # Base transformations
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Strong augmentation for contrastive views
    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load base datasets
    # train_dataset_base = torchvision.datasets.CIFAR10(
    #     root='/datasets/cv_datasets/data', train=True, download=True, transform=base_transform)
    train_dataset_base = torchvision.datasets.CIFAR10(
    root='/datasets/cv_datasets/data', train=True, download=True, transform=None)  # No transform here
    test_dataset = torchvision.datasets.CIFAR10(
        root='/datasets/cv_datasets/data', train=False, download=True, transform=base_transform)
    
    # Create contrastive dataset for training
    train_dataset = ContrastiveCIFAR10(train_dataset_base, contrastive_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

# Add noise to input for denoising training
def add_noise(inputs, noise_factor=0.2):
    noise = torch.randn_like(inputs) * noise_factor
    noisy_inputs = inputs + noise
    # Clip to ensure we stay in the valid range [-1, 1]
    return torch.clamp(noisy_inputs, -1., 1.)

# Training function with combined reconstruction and contrastive learning
def train_contrastive_encoder(
    model, 
    train_loader, 
    test_loader, 
    artifacts_dir, 
    num_epochs=50,  
    learning_rate=0.001, 
    noise_factor=0.2, 
    lambda_contrastive=0.5
):
    model = model.to(device)

    rec_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()  # <-- NEW
    contrastive_criterion = NTXentLoss(temperature=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_rec_losses = []
    train_contrastive_losses = []
    train_total_losses = []
    test_losses = []
    test_maes = []  # <-- NEW

    best_test_loss = float('inf')
    best_model_state = None

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_rec_loss = 0.0
        running_contrastive_loss = 0.0
        running_total_loss = 0.0

        for img1, img2, _ in train_loader:
            img1, img2 = img1.to(device), img2.to(device)
            noisy_img1 = add_noise(img1, noise_factor)
            noisy_img2 = add_noise(img2, noise_factor)

            optimizer.zero_grad()

            outputs1, _, proj1 = model(noisy_img1)
            outputs2, _, proj2 = model(noisy_img2)

            rec_loss1 = rec_criterion(outputs1, img1)
            rec_loss2 = rec_criterion(outputs2, img2)
            rec_loss = (rec_loss1 + rec_loss2) / 2

            contrastive_loss = contrastive_criterion(proj1, proj2)
            total_loss = rec_loss + lambda_contrastive * contrastive_loss

            total_loss.backward()
            optimizer.step()

            running_rec_loss += rec_loss.item()
            running_contrastive_loss += contrastive_loss.item()
            running_total_loss += total_loss.item()

        avg_rec_loss = running_rec_loss / len(train_loader)
        avg_contrastive_loss = running_contrastive_loss / len(train_loader)
        avg_total_loss = running_total_loss / len(train_loader)

        train_rec_losses.append(avg_rec_loss)
        train_contrastive_losses.append(avg_contrastive_loss)
        train_total_losses.append(avg_total_loss)

        # --- Validation ---
        model.eval()
        test_loss = 0.0
        test_mae = 0.0  # <-- NEW

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                noisy_inputs = add_noise(inputs, noise_factor)

                outputs, _, _ = model(noisy_inputs)
                test_loss += rec_criterion(outputs, inputs).item()
                test_mae += mae_criterion(outputs, inputs).item()  # <-- NEW

        avg_test_loss = test_loss / len(test_loader)
        avg_test_mae = test_mae / len(test_loader)  # <-- NEW
        test_losses.append(avg_test_loss)
        test_maes.append(avg_test_mae)  # <-- NEW

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, artifacts_dir / 'best_contrastive_encoder.pth')

        scheduler.step()

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Rec Loss: {avg_rec_loss:.4f}, '
              f'Contrastive Loss: {avg_contrastive_loss:.4f}, '
              f'Total Loss: {avg_total_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, '
              f'Test MAE: {avg_test_mae:.4f}, '  # <-- NEW
              f'LR: {scheduler.get_last_lr()[0]:.6f}, '
              f'Time: {elapsed_time:.1f}s')

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_rec_losses, label='Training Reconstruction Loss')
    plt.plot(train_contrastive_losses, label='Training Contrastive Loss')
    plt.plot(train_total_losses, label='Training Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label='Validation MSE Loss')
    plt.plot(test_maes, label='Validation MAE')  # <-- NEW
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss & MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(artifacts_dir / 'contrastive_encoder_loss.png')
    plt.close()

    return model, best_test_loss, avg_test_mae  # <-- RETURN MAE


# Classifier head remains the same
class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=10):
        super(ClassifierHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

def linear_evaluation(encoder, train_loader, test_loader, artifacts_dir, num_epochs=50, learning_rate=0.001):
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='/datasets/cv_datasets/data', train=True, download=True, transform=base_transform)
    
    train_loader_linear = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    classifier = ClassifierHead(embedding_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        classifier.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader_linear),
        epochs=num_epochs
    )
    
    train_accs = []
    test_accs = []
    best_acc = 0.0
    best_model_state = None
    
    # Precompute embeddings
    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []
    
    print("Precomputing embeddings...")
    with torch.no_grad():
        for data in train_loader_linear:
            inputs, labels = data
            inputs = inputs.to(device)
            embeddings = encoder.encode(inputs)
            train_embeddings.append(embeddings.cpu())
            train_labels.append(labels)
        
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            embeddings = encoder.encode(inputs)
            test_embeddings.append(embeddings.cpu())
            test_labels.append(labels)
    
    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_embeddings = torch.cat(test_embeddings, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    from torch.utils.data import TensorDataset
    
    train_dataset = TensorDataset(train_embeddings, train_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    
    train_loader_emb = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader_emb = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    for epoch in range(num_epochs):
        classifier.train()
        correct = 0
        total = 0
        
        for embeddings, labels in train_loader_emb:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # Validation
        classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in test_loader_emb:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = classifier(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = classifier.state_dict()
            torch.save(best_model_state, artifacts_dir / 'best_linear_classifier.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # Plot the accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Linear Evaluation Accuracy')
    plt.legend()
    plt.savefig(artifacts_dir / 'linear_eval_accuracy.png')
    plt.close()
    
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    return classifier, best_acc


# Visualization of embeddings with t-SNE
def visualize_embeddings(encoder, test_loader, artifacts_dir, num_samples=1000):
    encoder.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs = inputs.to(device)
            emb = encoder.encode(inputs)
            embeddings.append(emb.cpu().numpy())
            labels.append(target.numpy())
            
            if len(np.concatenate(embeddings)) >= num_samples:
                break
    
    embeddings = np.concatenate(embeddings)[:num_samples]
    labels = np.concatenate(labels)[:num_samples]
    
    # t-SNE dimensionality reduction
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot with colors based on class labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('t-SNE visualization of embeddings')
    plt.savefig(artifacts_dir / 'tsne_embeddings.png')
    plt.close()

# Main execution
def main(artifacts_dir):
    print(f"Using device: {device}")

    # Get data loaders for contrastive learning
    train_loader, test_loader = get_contrastive_data_loaders(batch_size=128)

    # Initialize the autoencoder with projection head
    embedding_dim = 128
    noise_factor = 0.15
    lambda_contrastive = 1.0  # Weight for contrastive loss

    model = CIFAR10Autoencoder(embedding_dim=embedding_dim)

    print(f"Training contrastive encoder with noise factor {noise_factor}...")

    model, best_test_loss, final_mae = train_contrastive_encoder(
        model, 
        train_loader, 
        test_loader, 
        artifacts_dir,
        num_epochs=50,  
        learning_rate=0.001,
        noise_factor=noise_factor,
        lambda_contrastive=lambda_contrastive
    )

    print(f"✅ Final test MSE loss: {best_test_loss:.4f}")
    print(f"✅ Final test MAE (mean absolute error): {final_mae:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), artifacts_dir / 'cifar10_contrastive_encoder.pth')

    # Visualize the learned embeddings
    print("Visualizing embeddings with t-SNE...")
    visualize_embeddings(model, test_loader, artifacts_dir)

    # Train classifier with linear evaluation protocol
    print("Performing linear evaluation on learned representations...")
    classifier, best_acc = linear_evaluation(
        model, 
        train_loader, 
        test_loader,
        artifacts_dir,
        num_epochs=50,
        learning_rate=0.003
    )

    print(f"🎯 Final test accuracy (linear eval): {best_acc:.2f}%")
    print("Done!")



#TO GENERATE LATENT SPACE VISUALIZATIONS

# from utils import plot_tsne
# from pathlib import Path

# class EncoderWrapper(nn.Module):
#     def __init__(self, autoencoder):
#         super().__init__()
#         self.autoencoder = autoencoder

#     def forward(self, x):
#         return self.autoencoder.encode(x)

# def main(artifacts_dir):
#     print(f"🧠 Skipping training, reloading saved contrastive model...")
#     print(f"Using device: {device}")

#     _, test_loader = get_contrastive_data_loaders(batch_size=128)

#     model = CIFAR10Autoencoder(embedding_dim=128)
#     model.load_state_dict(torch.load(Path(artifacts_dir) / 'best_contrastive_encoder.pth'))
#     model = model.to(device)
#     model.eval()

#     # Wrap encoder so it's compatible with plot_tsne
#     encoder = EncoderWrapper(model)

#     print("🌀 Generating t-SNE plots...")
#     plot_tsne(encoder, test_loader, device)
#     print("✅ t-SNE plots saved (image_tsne.png, latent_tsne.png)")
