import sys
from pathlib import Path
import h5py
import numpy as np

#sys.path.append(
#    "/gpfs01/berens/user/lschmors/Code/superior_colliculus/cne_timeseries/"
#)

def load_data_toy(
        filename_data,
        filename_labels,
):
    """
    Load toy data.

    Parameters:
        filename_data: path and name of data
        filename_labels: path and name of data


    Returns:
        data: [neurons, trials, time]
        labels: functional type
        type_names:
    """
    data_toy = np.load(filename_data).astype("float32")

    labels = np.load(filename_labels)

    len_type_names = np.unique(labels).shape[0]
    type_names = [str(i) for i in range(len_type_names + 1)]

    return data_toy, labels, type_names


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
    # df_clustered = pd.read_pickle(file_name)
    # labels = df_clustered["clusterID_sorted"].values.astype(int)
    labels = np.load(filepath + "/labels_bar_old.npy")

    # len_type_names = np.unique(labels).shape[0]
    # type_names = [str(i) for i in range(len_type_names + 1)]
    type_names = ['OFF', 'ON-OFF', 'ON', 'Sbc']

    return data_chirp_norm, data_bar_norm, labels, type_names


def load_data_allen(
        filepath="/gpfs01/berens/data/data/Allen_neuropixels_visual_coding/flashes_drifting_gratings/",
):
    """
    Load data from the Allen Institute dataset.
    Parameters:
        filepath: path to the data directory

    Returns:
        data_norm: normalized responses of shape (ROIs, trials, time)
        labels: functional type labels
        type_names: names of the functional types
    """
    # Load local chirp trial responses
    filepath = Path(filepath)
    data_flahses = np.load(filepath / "data_flashes_trials.npy").astype("float32")  # Shape: (ROIs, trials, time)
    data_gratings = np.load(filepath / "data_gratings_trials.npy").astype("float32")  # Shape: (ROIs, trials, time)

    # Normalize data
    data_flahses_norm = normalize_data(data_flahses)
    data_gratings_norm = normalize_data(data_gratings)

    # Load labels
    # labels = np.load(filepath / "labels_broad.npy")
    # type_names = ['Visual Cortex', 'Visual Midbrain', 'Visual Thalamus']
    labels = np.load(filepath / "labels.npy")
    type_names = [
        'APN',
        'LGd',
        'LGv',
        'LP',
        'VIS',
        'VISal',
        'VISam',
        'VISl',
        'VISp',
        'VISpm',
        'VISrl',
    ]

    return data_flahses_norm, data_gratings_norm, labels, type_names

def load_data_hcptrt(filepath="/gpfs01/berens/user/lschmors/Code/fMRI_TRACE/Data/data_EXTRACTED",
                     subject="sub-06"
):
    """
    Load human connectome project (HCP) test-retest (trt) task-fMRI responses for one subject.

    Reads the pre-extracted trial arrays for all HCP tasks (motor,
    gambling, working memory, social, emotion, language, relational), applies
    `normalize_data` to each, and loads the region labels shared across tasks.

    Parameters
    ----------
    filepath : str
        Root directory containing the `dataset_hcptrt_voxels_ca_parcels_<subject>`
        folders.
    subject : str
        Subject identifier, e.g. "sub-06".

    Returns
    -------
    data_*_norm : ndarray
        Seven normalized response arrays of shape [voxels, trials, time_bins],
        in the order motor, gambling, wm, social, emotion, language, relational.
    labels : ndarray
        Integer region label per voxel, indexing into `type_names`
        (taken from the motor run's region file).
    type_names : ndarray
        Unique region names, sorted.
    """
    # Load task responses [voxels, trials, time_bins]
    data_motor = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/motor_all_trials_norm.npy')
    data_gambling = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/gambling_all_trials_norm.npy')
    data_wm = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/wm_all_trials_norm.npy')
    data_social = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/social_all_trials_norm.npy')
    data_emotion = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/emotion_all_trials_norm.npy')
    data_language = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/language_all_trials_norm.npy')
    data_relational = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/relational_all_trials_norm.npy')

    # Normalize data
    data_motor_norm = normalize_data(data_motor)
    data_gambling_norm = normalize_data(data_gambling)
    data_wm_norm = normalize_data(data_wm)
    data_social_norm = normalize_data(data_social)
    data_emotion_norm = normalize_data(data_emotion)
    data_language_norm = normalize_data(data_language)
    data_relational_norm = normalize_data(data_relational)

    # Load labels
    y_arr_str = np.load(filepath + f"/dataset_hcptrt_voxels_ca_parcels_{subject}/motor_y_regions.npy")
    type_names, labels = np.unique(y_arr_str, return_inverse=True)

    return (data_motor_norm, data_gambling_norm, data_wm_norm, data_social_norm, data_emotion_norm, data_language_norm,
            data_relational_norm, labels, type_names)

def load_data_hcptrt_restingstate(filepath="/gpfs01/berens/user/lschmors/Code/fMRI_TRACE/Data/data_EXTRACTED",
                     subject="sub-06"
):
    """
    Load human connectome project (HCP) test-retest (trt) task-fMRI responses for one subject.

    Reads the pre-extracted trial arrays for all HCP tasks (motor,
    gambling, working memory, social, emotion, language, relational), applies
    `normalize_data` to each, and loads the region labels shared across tasks.

    Parameters
    ----------
    filepath : str
        Root directory containing the `dataset_hcptrt_voxels_ca_parcels_<subject>`
        folders.
    subject : str
        Subject identifier, e.g. "sub-06".

    Returns
    -------
    data_*_norm : ndarray
        Seven normalized response arrays of shape [voxels, trials, time_bins],
        in the order motor, gambling, wm, social, emotion, language, relational.
    labels : ndarray
        Integer region label per voxel, indexing into `type_names`
        (taken from the motor run's region file).
    type_names : ndarray
        Unique region names, sorted.
    """
    # Load task responses [voxels, trials, time_bins]
    data_restingstate = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/restingstate_all_trials_norm.npy')

    # Normalize data
    data_restingstate_norm = normalize_data(data_restingstate)

    # Load labels
    y_arr_str = np.load(filepath + f"/dataset_hcptrt_voxels_ca_parcels_{subject}/motor_y_regions.npy")
    type_names, labels = np.unique(y_arr_str, return_inverse=True)

    return (data_restingstate_norm, labels, type_names)


def load_data_hcptrt_task(filepath="/gpfs01/berens/user/lschmors/Code/fMRI_TRACE/Data/data_EXTRACTED",
                     subject="sub-06", task="motor",
):
    """
    Load human connectome project (HCP) test-retest (trt) task-fMRI responses for one subject.

    Reads the pre-extracted trial arrays for all HCP tasks (motor,
    gambling, working memory, social, emotion, language, relational), applies
    `normalize_data` to each, and loads the region labels shared across tasks.

    Parameters
    ----------
    filepath : str
        Root directory containing the `dataset_hcptrt_voxels_ca_parcels_<subject>`
        folders.
    subject : str
        Subject identifier, e.g. "sub-06".

    Returns
    -------
    data_*_norm : ndarray
        Seven normalized response arrays of shape [voxels, trials, time_bins],
        in the order motor, gambling, wm, social, emotion, language, relational.
    labels : ndarray
        Integer region label per voxel, indexing into `type_names`
        (taken from the motor run's region file).
    type_names : ndarray
        Unique region names, sorted.
    """
    # Load task responses [voxels, trials, time_bins]
    if task == "motor":
        data_task = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/motor_all_trials_norm.npy')
    if task == "gambling":
        data_task = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/gambling_all_trials_norm.npy')
    if task == "wm":
        data_task = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/wm_all_trials_norm.npy')
    if task == "social":
        data_task = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/social_all_trials_norm.npy')
    if task == "emotion":
        data_task = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/emotion_all_trials_norm.npy')
    if task == "language":
        data_task = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/language_all_trials_norm.npy')
    if task == "relational":
        data_task = np.load(f'{filepath}/dataset_hcptrt_voxels_ca_parcels_{subject}/relational_all_trials_norm.npy')
    # Normalize data
    data_task_norm = normalize_data(data_task)

    # Load labels
    y_arr_str = np.load(filepath + f"/dataset_hcptrt_voxels_ca_parcels_{subject}/motor_y_regions.npy")
    type_names, labels = np.unique(y_arr_str, return_inverse=True)

    return (data_task_norm, labels, type_names)


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
    # Replace zero standard deviations with a very small value to avoid division by zero
    unit_std = np.where(unit_std == 0, 1e-12, unit_std)

    # Normalize each unit's data across all trials using the grand mean and grand SD
    data_normalized = (data - unit_mean[:, np.newaxis, np.newaxis]) / unit_std[
        :, np.newaxis, np.newaxis
    ]

    return data_normalized

