import argparse
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from cne import ContrastiveEmbedding
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

sys.path.append(
    "/gpfs01/berens/user/lschmors/Code/superior_colliculus/cne_timeseries/"
)
from sc_utils import (
    ContrastiveTrialPairGenerator,
    TimeSeriesDataset,
    TimeSeriesMLP,
)

def load_data_toy(
    filepath="/gpfs01/berens/user/lschmors/Code/superior_colliculus/20241211_simple_toy_dataset/data/"
):
    """
    Load toy data.

    Parameters:
        filepath: path to data

    Returns:
        data: [ROIs, trials, time]
        labels: functional type
        type_names:

    """
    filepath = Path(filepath)
    data_toy = np.load(
            filepath / "toy_data.npy"
        ).astype("float32")

    labels = np.load(filepath / "toy_data_labels.npy")

    len_type_names = np.unique(labels).shape[0]
    type_names = [str(i) for i in range(len_type_names + 1)]

    return data_toy, labels, type_names

def load_data_bc(
    #filepath="/gpfs01/berens/data/data/BC_Franke2017_simulated_trials/",
    filepath='/gpfs01/berens/user/lschmors/Code/superior_colliculus/Data/BC_toy_data'
             '/bc_noise_data_2025_01_27/',
    trim=True,
):
    """
    Load data for local and global chirp responses.

    Parameters:
        filepath: path to bar and chrip response data
        trim: cut off first sec of chirp due to experimental session dependent adaptation phase

    Returns:
        data_local_chirp_norm: normalized local chirp response data
        data_global_chirp_norm: normalized global chirp response data
        labels: functional type

    """
    # Load local chirp trial responses
    filepath = Path(filepath)
    fchirp = np.load(filepath / "bc_local_chirp_noise_0.npy").astype("float32")
    labels = np.load(filepath / "bc_local_chirp_noise_0_labels.npy")

    # Normalize data
    data_local_chirp_norm = normalize_data(fchirp)

    if trim:
        # Trim first second of chirp
        data_local_chirp_norm = data_local_chirp_norm[:, :, 9:]

    type_names = [
        "1",
        "2",
        "3a",
        "3b",
        "4",
        "5t",
        "5o",
        "5i",
        "X",
        "6",
        "7",
        "8",
        "9",
        "R",
    ]

    return data_local_chirp_norm, labels, type_names


def load_data_sc(
    filepath="/gpfs01/berens/data/data/superior_colliculus",
    trim=True,
    flatten_bar=False,
):
    """
    Load data for chirp responses and bar responses.

    Parameters:
        filepath: path to bar and chrip response data
        trim: cut off first sec of chirp due to experimental session dependent adaptation phase
        flatten_bar: if True, flatten bar responses (8 directions) to [ROIs, trials, 8*2sec]

    Returns:
        data_chirp_norm: normalized chirp response data
        data_bar_norm: normalized bar response data
        labels: functional type

    """
    # Load chirp trial responses
    file_name = filepath + "/chirp_trials.h5"
    with h5py.File(file_name, "r") as file:
        # Check if the dataset exists
        if "chirp_trials" in file:
            data_chirp = file["chirp_trials"][:]  # Shape: (ROIs, trials, time)
        else:
            raise ValueError("Dataset 'chirp_trials' not found in file.")

    # Load bar trial responses
    file_name_bar = filepath + "/bar_trials.h5"
    with h5py.File(file_name_bar, "r") as file:
        # Check if the dataset exists
        if "bar_trials" in file:
            data_bar = file["bar_trials"][:]  # Shape: (ROIs, trials, time)
        else:
            raise ValueError("Dataset 'bar_trials' not found in file.")
    if flatten_bar == False:
        # Get temporal response only
        data_bar_non_norm = np.mean(data_bar, axis=2)  # [ROIs, trials, 2sec]
    else:
        data_bar_non_norm = data_bar.reshape(
            [
                data_bar.shape[0],
                data_bar.shape[1],
                data_bar.shape[2] * data_bar.shape[3],
            ]
        )  # [ROIs, trials, 8*2sec]
    # Normalize data
    data_chirp_norm = normalize_data(data_chirp)
    data_bar_norm = normalize_data(data_bar_non_norm)

    if trim:
        # Trim first second of chirp
        data_chirp_norm = data_chirp_norm[:, :, 9:]

    # Load labels
    file_name = (
        filepath + "/20240207_df_clusterd_identified.pkl"
    )
    #df_clustered = pd.read_pickle(file_name)
    #labels = df_clustered["clusterID_sorted"].values.astype(int)
    labels = np.load(filepath + "/labels_bar.npy")

    #len_type_names = np.unique(labels).shape[0]
    #type_names = [str(i) for i in range(len_type_names + 1)]
    type_names = ['OFF', 'ON-OFF', 'ON', 'Sbc']

    return data_chirp_norm, data_bar_norm, labels, type_names


def normalize_data(data):
    """
    Normalizes the single trial responses per ROI.

    Parameters:
    data (numpy array): The input data array of shape (samples, trials, time).

    Returns:
    data_normalized (numpy array): The normalized data array where every single trial
        response is normalized using the mean and SD of the mean response across all trials.
    """

    # Calculate the mean and standard deviation per unit across all trials
    unit_average = np.mean(data, axis=1)
    unit_mean = np.mean(unit_average, axis=1)
    unit_std = np.std(unit_average, axis=1)

    # Normalize each unit's data across all trials using the grand mean and grand SD
    data_normalized = (data - unit_mean[:, np.newaxis, np.newaxis]) / unit_std[
        :, np.newaxis, np.newaxis
    ]

    return data_normalized

