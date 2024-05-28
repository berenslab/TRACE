import argparse
from datetime import datetime
import h5py
import os
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from cne import ContrastiveEmbedding
sys.path.append("/gpfs01/berens/user/lschmors/Code/superior_colliculus/cne_timeseries/")
from sc_utils import ContrastiveTrialPairGenerator, TimeSeriesDataset, TimeSeriesMLP

def load_data(filepath='/gpfs01/berens/user/lschmors/Code/superior_colliculus'):
    # Load chirp trial responses
    file_name = filepath + '/Data/retinal_axons/chirp_trials.h5'
    with h5py.File(file_name, 'r') as file:
        # Check if the dataset exists
        if "chirp_trials" in file:
            data_chirp = file["chirp_trials"][:]  # Shape: (ROIs, trials, time)
        else:
            raise ValueError("Dataset 'chirp_trials' not found in file.")

    # Load labels
    file_name = filepath + '/20240207_tSNE/data/20240207_df_clusterd_identified.pkl'
    df_clustered = pd.read_pickle(file_name)
    labels = df_clustered['clusterID_sorted'].values.astype(int)

    # Normalize data
    # Calculate the mean and std per neuron across all trials
    chirp_average = np.mean(data_chirp, axis=1)
    chirp_mean_per_neuron = np.mean(chirp_average, axis=1)
    chirp_std_per_neuron = np.std(chirp_average, axis=1)
    # Normalize single trial responses using the grand mean and grand SD over all trials
    # Normalize per neuron
    data_chirp_norm = ((data_chirp - chirp_mean_per_neuron[:, np.newaxis, np.newaxis]) /
                                            chirp_std_per_neuron[:, np.newaxis, np.newaxis])
    data = data_chirp_norm
    return data, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default='TimeSeriesMLP')
    parser.add_argument("-r", "--run", type=int, help="Seed for random number generator.")
    parser.add_argument("-e", "--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("-d", "--dir", type=str, help="Directory to save results.")
    parser.add_argument("-o", "--output_dim", type=int, help="Dimensions of final layer.")
    parser.add_argument("-l", "--loss_mode", type=str, default='infonce', help="Mode of the loss.")
    parser.add_argument("-t", "--trials", type=int, default=5, help="Number of trials to average over for generating positive pair.")
    args = parser.parse_args()

    # Load data and create dataset
    data, labels = load_data()
    dataset = ContrastiveTrialPairGenerator(data, n_trials_pos_pair=int(args.trials))

    # Set parameters
    model_name = str(args.model_name)

    # DataLoader params
    run = int(args.run)  # serves as seed
    gen = torch.Generator().manual_seed(run)
    num_workers = 8

    batch_size = int(args.batch_size) #1024
    negative_samples = 10  # example, adjust based on your requirements

    # ContrastiveEmbedding params
    n_epochs = int(args.epochs)
    device = "cuda:0"
    learning_rate = 0.03 * batch_size / 256
    lr_min_factor = 0.0
    momentum = 0.0
    temperature = 0.5
    eps = 1.0
    clamp_high = float("inf")
    clamp_low = float("-inf")
    Z = 1.0
    loss_mode = args.loss_mode  # or infonce, nce, neg_sample, ...
    metric = "euclidean"
    optimizer = "sgd"
    weight_decay = 5e-4
    anneal_lr = "cosine"
    lr_decay_rate = 0.1
    clip_grad = True
    save_freq = 25
    callback = None
    print_freq_epoch = "auto"
    print_freq_iteration = None
    seed = 0
    loss_aggregation = "sum"
    warmup_epochs = 5
    warmup_lr = 0
    early_exaggeration = True
    output_dim = int(args.output_dim)

    # Saving directory
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_name = f"{datetime_string}_embd_{model_name}_epochs{n_epochs}_batchsize{batch_size}" \
                f"_outputdim{output_dim}_run{run}_ntrials{int(args.trials)}"
    print(f'Directory: {args.dir}')
    print(f'File name: {file_name}')
    plots_dir = os.path.join(args.dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    models_dir = os.path.join(args.dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Create DataLoader and initialize model
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=gen,
        drop_last=True,
    )
    model = TimeSeriesMLP(input_features=data.shape[2], n_features=output_dim)
    cne = ContrastiveEmbedding(
        model=model,
        batch_size=batch_size,
        negative_samples=negative_samples,  # or 2 * batch_size
        n_epochs=n_epochs,
        device=device,
        learning_rate=learning_rate,
        lr_min_factor=lr_min_factor,
        momentum=momentum,
        temperature=temperature,
        eps=eps,
        clamp_high=clamp_high,
        clamp_low=clamp_low,
        Z=Z,
        loss_mode=loss_mode,  # or infonce, nce, neg_sample, ...
        metric=metric,
        optimizer=optimizer,
        weight_decay=weight_decay,
        anneal_lr=anneal_lr,
        lr_decay_rate=lr_decay_rate,
        clip_grad=clip_grad,
        save_freq=save_freq,
        callback=callback,
        print_freq_epoch=print_freq_epoch,
        print_freq_iteration=print_freq_iteration,
        seed=seed,
        loss_aggregation=loss_aggregation,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        early_exaggeration=early_exaggeration,
    )

    # Fit model
    start = time.time()
    cne.fit_transform(loader)
    cne.time = time.time() - start
    print(f"Time: {cne.time / 60} mins")

    # Save model
    filepath = os.path.join(models_dir, f"{file_name}.pkl")
    with open(filepath, "wb") as file:
        pickle.dump(cne, file, pickle.HIGHEST_PROTOCOL)

    # Save loss plot
    loss_mean_per_epoch = []
    for epoch_idx in range(n_epochs):
        loss_mean_per_epoch.append(np.mean(cne.losses[epoch_idx]))
    loss_mean_per_epoch = np.array(loss_mean_per_epoch)
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(loss_mean_per_epoch)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    sns.despine()
    filepath = os.path.join(plots_dir, f"{file_name}_loss_over_epochs.png")
    fig.patch.set_facecolor('white')
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(),
                transparent=False)

    # Save embeddings
    dataset_embd = TimeSeriesDataset(data=data)
    loader_embd = DataLoader(
        dataset_embd,
        shuffle=False,
        batch_size=batch_size,  # Adjust based on your requirements
        num_workers=num_workers,  # Adjust based on your system capabilities
        drop_last=False,
    )
    embedding = []
    cne.model.eval()
    with torch.no_grad():
        for batch in loader_embd:
            input_data = batch
            if isinstance(input_data, tuple):
                input_data = input_data[0]  # Handle case when dataset returns (data, label)
            # Ensure input_data has batch and channel dimensions
            if input_data.ndim == 2:
                input_data = input_data.unsqueeze(1)  # Add channel dimension
            input_data = input_data.to('cuda')
            z = cne.model(input_data)
            embedding.extend(z.cpu().numpy())  # Append embeddings
    embedding = np.array(embedding)
    # Plotting
    n_clusters = 50
    cmap = ListedColormap(sns.husl_palette(n_clusters).as_hex())
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=5)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    sns.despine(bottom=True, left=True)
    filepath = os.path.join(plots_dir, f"{file_name}_embedding.png")
    fig.patch.set_facecolor('white')
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(),
                transparent=False)

if __name__ == "__main__":
    main()
