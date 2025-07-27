import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from torch.utils.data import random_split
from utils import plot_tsne

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder architecture
class CIFAR10Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CIFAR10Encoder, self).__init__()
        
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
    
    def forward(self, x):
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

# Classifier head
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

# Combined model: Encoder + Classifier
class EncoderClassifier(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=10):
        super(EncoderClassifier, self).__init__()
        self.encoder = CIFAR10Encoder(embedding_dim)
        self.classifier = ClassifierHead(embedding_dim, num_classes)
    
    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding

# Data loading with augmentation
def get_augmented_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root='/datasets/cv_datasets/data', train=True, download=True, transform=transform)

    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root='/datasets/cv_datasets/data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

# Training function for end-to-end Encoder+Classifier
def train_encoder_classifier(model, train_loader, val_loader, artifacts_dir, num_epochs=100, learning_rate=0.001):
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )
    
    train_accs = []
    val_accs = []              # <-- NEW
    train_losses = []
    val_losses = []            # <-- NEW
    
    best_acc = 0.0
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # --- Validation loop (REPLACED old test loop) ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                logits, _ = model(inputs)
                loss = criterion(logits, labels)
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = 100 * correct / total
        val_accs.append(val_acc)
        # --------------------------------------------------

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, artifacts_dir / 'best_encoder_classifier.pth')
        
        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}, '
              f'Time: {elapsed_time:.1f}s')

    # --- Plot training vs. validation ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')  # <-- updated
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')    # <-- updated
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(artifacts_dir / 'encoder_classifier_performance.png')
    plt.close()

    print(f"Best validation accuracy: {best_acc:.2f}%")
    return model, best_acc

# Main execution
def main(artifacts_dir):
    print(f"Using device: {device}")

    # Updated to get val_loader as well
    train_loader, val_loader, test_loader = get_augmented_data_loaders(batch_size=128)

    model = EncoderClassifier(embedding_dim=128)
    print("Training encoder+classifier model...")

    model, best_acc = train_encoder_classifier(
        model,
        train_loader,
        val_loader,        # Correct position
        artifacts_dir,     # This should now be 4th
        num_epochs=100,
        learning_rate=0.001
    )


    print(f"Final best validation accuracy: {best_acc:.2f}%")
    
    # 🔍 Evaluate best model on test set
    print("Evaluating best model on test set...")
    best_model = EncoderClassifier(embedding_dim=128)
    best_model.load_state_dict(torch.load(artifacts_dir / 'best_encoder_classifier.pth'))
    best_model = best_model.to(device)
    best_model.eval()
    plot_tsne(best_model.encoder, test_loader, device)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = best_model(inputs)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_test_acc = 100 * correct / total
    print(f"🎯 Final test accuracy: {final_test_acc:.2f}%")

    print("Done!")

# from utils import plot_tsne  # make sure this is already there

# def main(artifacts_dir):
#     print(f"Using device: {device}")

#     # Only load test set — we’re just doing t-SNE
#     _, _, test_loader = get_augmented_data_loaders(batch_size=128)

#     # Load trained encoder model (from best checkpoint)
#     model = EncoderClassifier(embedding_dim=128)
#     model.load_state_dict(torch.load(artifacts_dir / 'best_encoder_classifier.pth'))
#     model = model.to(device)
#     model.eval()

#     # Just call the supplied t-SNE function — it saves plots automatically
#     print("Running t-SNE on latent and image space...")
#     plot_tsne(model.encoder, test_loader, device)

#     print("✅ t-SNE plots saved as 'latent_tsne.png' and 'image_tsne.png' in current directory.")


