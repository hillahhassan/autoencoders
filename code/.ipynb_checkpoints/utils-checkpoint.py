import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# def plot_tsne(model, dataloader, device):
#     '''
#     model - torch.nn.Module subclass. This is your encoder model
#     dataloader - test dataloader to over over data for which you wish to compute projections
#     device - cuda or cpu (as a string)
#     '''
#     model.eval()
    
#     images_list = []
#     labels_list = []
#     latent_list = []
    
#     with torch.no_grad():
#         for data in dataloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
            
#             #approximate the latent space from data
#             latent_vector = model(images)
            
#             images_list.append(images.cpu().numpy())
#             labels_list.append(labels.cpu().numpy())
#             latent_list.append(latent_vector.cpu().numpy())
    
#     images = np.concatenate(images_list, axis=0)
#     labels = np.concatenate(labels_list, axis=0)
#     latent_vectors = np.concatenate(latent_list, axis=0)
    
#     # Plot TSNE for latent space
#     tsne_latent = TSNE(n_components=2, random_state=0)
#     latent_tsne = tsne_latent.fit_transform(latent_vectors)
    
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)  # Smaller points
#     plt.colorbar(scatter)
#     plt.title('t-SNE of Latent Space')
#     plt.savefig('latent_tsne.png')
#     plt.close()
    
#     #plot image domain tsne
#     tsne_image = TSNE(n_components=2, random_state=42)
#     images_flattened = images.reshape(images.shape[0], -1)
#     image_tsne = tsne_image.fit_transform(images_flattened)
    
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)  
#     plt.colorbar(scatter)
#     plt.title('t-SNE of Image Space')
#     plt.savefig('image_tsne.png')
#     plt.close()
    
#my addittions:
def visualize_reconstruction(model, dataloader, device, num_images=5):
    model.to(device)
    model.eval()

    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images[:num_images].to(device)

        reconstructed, _ = model(images)

        # Convert tensors to numpy
        images = images.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, H, W) → (N, H, W, C)
        reconstructed = reconstructed.cpu().numpy().transpose(0, 2, 3, 1)

        fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
        
        for i in range(num_images):
            axes[0, i].imshow(images[i].squeeze(), cmap="gray" if images.shape[-1] == 1 else None)
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray" if images.shape[-1] == 1 else None)
            axes[1, i].axis("off")

        plt.suptitle("Top: Original Images | Bottom: Reconstructed Images")
        plt.show()
        
#MODIFIED VERSION TO ACCEPT PATH
def plot_tsne(model, dataloader, device, latent_path="latent_tsne.png", image_path="image_tsne.png"):
    '''
    model - torch.nn.Module subclass. This is your encoder model
    dataloader - test dataloader to go over data for which you wish to compute projections
    device - cuda or cpu
    latent_path - path to save the latent space t-SNE plot
    image_path - path to save the image space t-SNE plot
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()

    images_list = []
    labels_list = []
    latent_list = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            latent_vector = model(images)

            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)

    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.savefig(latent_path)
    plt.close()

    # Plot TSNE for image space
    tsne_image = TSNE(n_components=2, random_state=42)
    images_flattened = images.reshape(images.shape[0], -1)
    image_tsne = tsne_image.fit_transform(images_flattened)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Space')
    plt.savefig(image_path)
    plt.close()
