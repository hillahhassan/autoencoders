import argparse
import random
import numpy as np
import torch
from pathlib import Path

import mnist
import cifar10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')

    # Dataset argument: now allow "mnist", "cifar10", or "all"
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['mnist', 'cifar10', 'all'], 
        default='all',
        help='Which dataset to use: mnist, cifar10, or all for both'
    )

    # Training task argument: now allow a new option "all" to run every task for a dataset
    parser.add_argument(
        '--training-task',
        type=str,
        choices=['autoencoder', 'classification', 'contrastive', 'all'],
        default='all',
        help=(
            "Select which training task to run:\n"
            " - autoencoder: Train autoencoder with reconstruction, then freeze and train classifier.\n"
            " - classification: Train encoder and classifier together with classification loss.\n"
            " - contrastive: Train an encoder using contrastive learning, then freeze and train classifier.\n"
            " - all: Run all training tasks"
        )
    )
    return parser.parse_args()

def run_task(dataset, task, base_dir):
    # Create the task directory: artifacts/<dataset>/<training-task>
    task_dir = base_dir / dataset / task
    task_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running {task} for {dataset} at {task_dir}")
    
    if dataset == "mnist":
        if task == "autoencoder":
            mnist.autoencoder.main(task_dir)
            mnist.autoencoder_classifier.main(task_dir)
        elif task == "classification":
            mnist.encoder_classifier.main(task_dir)
        elif task == "contrastive":
            mnist.contrastive_encoder_classifier.main(task_dir)
            # mnist.contrastive_encoder_classifier.plot_tsne_main(task_dir)
    elif dataset == "cifar10":
        if task == "autoencoder":
            cifar10.autoencoder_classifier.main(task_dir)
        elif task == "classification":
            cifar10.encoder_classifier.main(task_dir)
        elif task == "contrastive":
            cifar10.contrastive_autoencoder_classifier.main(task_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)
    
    base_dir = Path("artifacts")
    
    # Determine which datasets and tasks to run
    datasets = [args.dataset] if args.dataset != "all" else ["mnist", "cifar10"]
    tasks = [args.training_task] if args.training_task != "all" else ["autoencoder", "classification", "contrastive"]
    
    print("Starting training runs with the following configuration:")
    print(f"Datasets: {datasets}")
    print(f"Training Tasks: {tasks}")
    
    # Loop over the selected datasets and tasks, and run each one
    for dataset in datasets:
        for task in tasks:
            run_task(dataset, task, base_dir)