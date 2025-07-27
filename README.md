# Autoencoders

This project explores various methods for training autoencoders on the CIFAR-10 dataset, including contrastive learning and denoising objectives. We evaluate the quality of learned representations using reconstruction loss, classification performance, and latent space visualization.

## Files

- `2367801_Final_Project_Winter_25.pdf`  
  📄 Contains the final project description and instructions provided for this assignment.

- `wet.pdf`  
  📄 A summary of the performance of each autoencoder variant trained in this project.

- `general_questions.pdf`  
  📄 A collection of deep learning theory questions and answers. These are not necessarily tied directly to this project.

- `code/`  
  📁 Contains all source code used for training and evaluating the autoencoders, including:
  - Encoder architecture with projection head
  - Denoising + contrastive training loop
  - Linear evaluation protocol
  - t-SNE visualization

## Highlights

- Contrastive encoder with NT-Xent loss
- Denoising autoencoder training
- Linear evaluation on learned embeddings
- Visualization with t-SNE
- MAE and MSE evaluation

---

Built for deep representation learning experiments in the context of an advanced deep learning final project.
