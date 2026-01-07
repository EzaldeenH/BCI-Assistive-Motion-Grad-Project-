import tensorflow as tf
import numpy as np
import os
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test

def to_one_hot(y, by_sub=False):
    if by_sub:
        new_array = np.array(["nan" for nan in range(len(y))])
        for index, label in enumerate(y):
            new_array[index] = ''.join([i for i in label if not i.isdigit()])
    else:
        new_array = y.copy()
    total_labels = np.unique(new_array)
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x
    for x in range(len(new_array)):
        new_array[x] = mapping[new_array[x]]

    return tf.keras.utils.to_categorical(new_array)

def load_subject_data(subject: int, data_path: str, exclude_base: bool = False):
    """
    Given a subject number (@subject) and the original dataset
    path (@data_path), this function returns:
        xs: The time series; a numpy array of shape (n_sample, 64, 641)
        y: The labels, a list of length n_samples
        ch_names: The 64 channels order in the xs array
    """
    runs = [4, 6, 8, 10, 12, 14]
    task2 = [4, 8, 12]
    task4 = [6, 10, 14]
    if len(str(subject)) == 1:
        sub_name = "S" + "00" + str(subject)
    elif len(str(subject)) == 2:
        sub_name = "S" + "0" + str(subject)
    else:
        sub_name = "S" + str(subject)
    sub_folder = os.path.join(data_path, sub_name)
    subject_runs = []
    for run in runs:
        if len(str(run)) == 1:
            path_run = os.path.join(sub_folder,
                                    sub_name + "R" + "0" + str(run) + ".edf")
        else:
            path_run = os.path.join(sub_folder,
                                    sub_name + "R" + str(run) + ".edf")
        raw_run = read_raw_edf(path_run, preload=True)
        len_run = np.sum(
            raw_run._annotations.duration)
        if len_run > 124:
            raw_run.crop(tmax=124)

        """
        B indicates baseline
        L indicates motor imagination of opening and closing left fist;
        R indicates motor imagination of opening and closing right fist;
        LR indicates motor imagination of opening and closing both fists;
        F indicates motor imagination of opening and closing both feet.
        """

        if int(run) in task2:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "L"
                if an == "T2":
                    raw_run.annotations.description[index] = "R"
        if int(run) in task4:
            for index, an in enumerate(raw_run.annotations.description):
                if an == "T0":
                    raw_run.annotations.description[index] = "B"
                if an == "T1":
                    raw_run.annotations.description[index] = "LR"
                if an == "T2":
                    raw_run.annotations.description[index] = "F"
        subject_runs.append(raw_run)
    raw_conc = concatenate_raws(subject_runs)
    indexes = []
    for index, value in enumerate(raw_conc.annotations.description):
        if value == "BAD boundary" or value == "EDGE boundary":
            indexes.append(index)
    raw_conc.annotations.delete(indexes)

    # Apply bandpass filter (4-60 Hz) as requested
    raw_conc.filter(l_freq=4., h_freq=60., fir_design='firwin', skip_by_annotation='edge')
    #
    # # Apply notch filter (60 Hz) for USA power line noise
    # raw_conc.notch_filter(freqs=60., fir_design='firwin', skip_by_annotation='edge')

    eegbci.standardize(raw_conc)
    montage = make_standard_montage('standard_1005')
    raw_conc.set_montage(montage)
    tmin = 0
    tmax = 4
    if exclude_base:
        event_id = dict(F=2, L=3, LR=4, R=5)#4class
        # event_id = dict(L=3,R=5)#2class

    else:
        event_id = dict(B=1, F=2, L=3, LR=4, R=5)
    events, _ = mne.events_from_annotations(raw_conc, event_id=event_id)

    #64
    picks = mne.pick_types(raw_conc.info, meg=False, eeg=True, stim=False,
                           eog=False, exclude='bads')
    #22
    # picks = mne.pick_types(raw_conc.info, meg=False, eeg=False, stim=False,
    #                        eog=False,include=["Fz", "FC3","FC1", "FCz","FC2", "FC4","C5",  "C3","C1",  "Cz",
    #            "C2",  "C4","C6", "CP3","CP1", "CPz","CP2", "CP4","P1", "Pz","P2","POz"],exclude='bads')

    # 18
    # picks = mne.pick_types(raw_conc.info, meg=False, eeg=False, stim=False,
    #                        eog=False,include=["FC5","FC3","FC1","FC2", "FC4","FC6","C5","C3","C1",
    #            "C2","C4","C6","CP5","CP3","CP1", "CP2", "CP4", "CP6"],exclude='bads')

    #12
    # picks = mne.pick_types(raw_conc.info, meg=False, eeg=False, stim=False,
    #                        eog=False, include=["FC3", "FC1", "FC2", "FC4","C3", "C1","C2", "C4","CP3", "CP1","CP2", "CP4"], exclude='bads')

    epochs = Epochs(raw_conc, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)

    print(epochs[0].ch_names)

    y = list()
    for index, data in enumerate(epochs):
        y.append(epochs[index]._name)

    xs = np.array([epoch for epoch in epochs])

    return xs, y, raw_conc.ch_names

def load_physionet(path):
    exclude = [38, 88, 89, 92, 100, 104]
    subjects = [n for n in np.arange(1, 110) if n not in exclude]

    xs = list()
    ys = list()
    data_x = list()
    data_y = list()
    for subject in subjects:
        x, y, ch_names = load_subject_data(subject,path,True)
        print(x.shape)
        xs.append(x)
        ys.append(y)
    data_x = np.concatenate(xs)
    data_y = np.concatenate(ys)

    N_tr, N_ch, _ = data_x.shape
    data_x = data_x[:, :, :640].reshape(N_tr, 1, N_ch, -1)
    y_one_hot  = to_one_hot(data_y, by_sub=False)

    # Create Validation/test
    x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(data_x,
                                                                                y_one_hot,
                                                                                stratify=y_one_hot,
                                                                                test_size=0.10,
                                                                                random_state=42)

    #Scale indipendently train/test
    x_train_scaled_raw, x_test_valid_scaled_raw = standardize_data(x_train_raw, x_valid_test_raw, N_ch)

    print(x_train_scaled_raw.shape, x_test_valid_scaled_raw.shape)
    return x_train_scaled_raw,y_train_raw,x_test_valid_scaled_raw,y_valid_test_raw


def load_physionet_raw(path):
    """Load all Physionet data without train/test split for k-fold cross-validation.
    
    Returns:
        data_x: numpy array of shape (N_trials, 1, N_channels, N_timepoints)
        y_one_hot: one-hot encoded labels
        y_labels: raw class labels (integers) for stratification
    """
    exclude = [38, 88, 89, 92, 100, 104]
    subjects = [n for n in np.arange(1, 110) if n not in exclude]

    xs = list()
    ys = list()
    for subject in subjects:
        x, y, ch_names = load_subject_data(subject, path, True)
        print(f"Subject {subject}: {x.shape}")
        xs.append(x)
        ys.append(y)
    data_x = np.concatenate(xs)
    data_y = np.concatenate(ys)

    N_tr, N_ch, _ = data_x.shape
    data_x = data_x[:, :, :640].reshape(N_tr, 1, N_ch, -1)
    y_one_hot = to_one_hot(data_y, by_sub=False)
    
    # Get integer labels for stratification
    y_labels = np.argmax(y_one_hot, axis=1)

    print(f"Total data shape: {data_x.shape}, Labels shape: {y_one_hot.shape}")
    return data_x, y_one_hot, y_labels, N_ch


# ============================================================================
# TCPL-Specific Data Loading Functions
# ============================================================================

def load_physionet_by_subject(path, n_timepoints=640):
    """
    Load PhysioNet data organized by subject for meta-learning.
    
    Returns a dictionary mapping subject_id -> (X, y_onehot, y_labels)
    This allows episodic sampling where we treat each subject as a "task".
    
    Parameters
    ----------
    path : str
        Path to PhysioNet dataset
    n_timepoints : int
        Number of time points to keep (default: 640)
        
    Returns
    -------
    subject_data : dict
        {subject_id: {'X': array, 'y_onehot': array, 'y_labels': array}}
    n_channels : int
        Number of EEG channels
    n_classes : int
        Number of classes
    """
    exclude = [38, 88, 89, 92, 100, 104]
    subjects = [n for n in np.arange(1, 110) if n not in exclude]
    
    subject_data = {}
    n_channels = None
    
    for subject in subjects:
        x, y, ch_names = load_subject_data(subject, path, True)
        
        N_tr, N_ch, _ = x.shape
        if n_channels is None:
            n_channels = N_ch
            
        # Reshape to standard format: (trials, 1, channels, timepoints)
        x = x[:, :, :n_timepoints].reshape(N_tr, 1, N_ch, -1)
        
        # Convert labels
        y_onehot = to_one_hot(np.array(y), by_sub=False)
        y_labels = np.argmax(y_onehot, axis=1)
        
        # Z-score normalize within subject
        x = standardize_subject_data(x, N_ch)
        
        subject_data[subject] = {
            'X': x,
            'y_onehot': y_onehot,
            'y_labels': y_labels
        }
        print(f"Subject {subject}: {x.shape}, classes: {np.bincount(y_labels)}")
    
    n_classes = y_onehot.shape[1]
    print(f"\nLoaded {len(subject_data)} subjects, {n_channels} channels, {n_classes} classes")
    
    return subject_data, n_channels, n_classes


def standardize_subject_data(X, n_channels):
    """
    Z-score normalize data within a single subject.
    
    Parameters
    ----------
    X : ndarray
        Shape (n_trials, 1, n_channels, n_timepoints)
    n_channels : int
        Number of channels
        
    Returns
    -------
    X_normalized : ndarray
        Z-score normalized data
    """
    X_norm = X.copy()
    for ch in range(n_channels):
        # Get all time points across all trials for this channel
        ch_data = X_norm[:, 0, ch, :]
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        if std > 0:
            X_norm[:, 0, ch, :] = (ch_data - mean) / std
    return X_norm


class EpisodicDataGenerator:
    """
    Episodic data generator for TCPL meta-learning.
    
    Each episode consists of:
    - Support set: n_shot samples per class from one subject
    - Query set: remaining samples from the same subject
    
    This simulates the few-shot adaptation scenario during training.
    
    Parameters
    ----------
    subject_data : dict
        Output from load_physionet_by_subject()
    n_shot : int
        Number of samples per class in support set
    n_query : int or None
        Number of samples per class in query set (None = use all remaining)
    seed : int
        Random seed for reproducibility
    """
    
    def __init__(self, subject_data, n_shot=5, n_query=None, seed=42):
        self.subject_data = subject_data
        self.subject_ids = list(subject_data.keys())
        self.n_shot = n_shot
        self.n_query = n_query
        self.rng = np.random.RandomState(seed)
        
        # Determine number of classes from first subject
        first_subj = self.subject_data[self.subject_ids[0]]
        self.n_classes = first_subj['y_onehot'].shape[1]
        
    def sample_episode(self, subject_id=None):
        """
        Sample one episode (support + query set) from a subject.
        
        Parameters
        ----------
        subject_id : int or None
            Specific subject to sample from (None = random)
            
        Returns
        -------
        support_X : ndarray
            Shape (n_shot * n_classes, 1, C, T)
        support_y : ndarray
            One-hot labels for support set
        query_X : ndarray
            Shape (n_query * n_classes, 1, C, T) 
        query_y : ndarray
            One-hot labels for query set
        """
        if subject_id is None:
            subject_id = self.rng.choice(self.subject_ids)
            
        data = self.subject_data[subject_id]
        X = data['X']
        y_onehot = data['y_onehot']
        y_labels = data['y_labels']
        
        support_X, support_y = [], []
        query_X, query_y = [], []
        
        for class_idx in range(self.n_classes):
            # Get all samples for this class
            class_mask = y_labels == class_idx
            class_X = X[class_mask]
            class_y = y_onehot[class_mask]
            
            n_samples = len(class_X)
            if n_samples < self.n_shot + 1:
                # Not enough samples, use all but one for support
                n_support = max(1, n_samples - 1)
            else:
                n_support = self.n_shot
            
            # Random permutation
            perm = self.rng.permutation(n_samples)
            
            # Split into support and query
            support_idx = perm[:n_support]
            query_idx = perm[n_support:]
            
            if self.n_query is not None:
                query_idx = query_idx[:self.n_query]
            
            support_X.append(class_X[support_idx])
            support_y.append(class_y[support_idx])
            query_X.append(class_X[query_idx])
            query_y.append(class_y[query_idx])
        
        # Concatenate all classes
        support_X = np.concatenate(support_X, axis=0)
        support_y = np.concatenate(support_y, axis=0)
        query_X = np.concatenate(query_X, axis=0)
        query_y = np.concatenate(query_y, axis=0)
        
        # Shuffle
        support_perm = self.rng.permutation(len(support_X))
        query_perm = self.rng.permutation(len(query_X))
        
        return (support_X[support_perm], support_y[support_perm],
                query_X[query_perm], query_y[query_perm])
    
    def generate_episodes(self, n_episodes, subjects=None):
        """
        Generator that yields multiple episodes.
        
        Parameters
        ----------
        n_episodes : int
            Number of episodes to generate
        subjects : list or None
            List of subject IDs to sample from (None = all)
            
        Yields
        ------
        support_X, support_y, query_X, query_y
        """
        if subjects is None:
            subjects = self.subject_ids
            
        for _ in range(n_episodes):
            subject_id = self.rng.choice(subjects)
            yield self.sample_episode(subject_id)
    
    def reset_seed(self, seed):
        """Reset random state for reproducibility."""
        self.rng = np.random.RandomState(seed)


def split_subjects_kfold(subject_ids, n_folds=10, seed=42):
    """
    Split subjects into k folds for cross-validation.
    
    Parameters
    ----------
    subject_ids : list
        List of subject IDs
    n_folds : int
        Number of folds
    seed : int
        Random seed
        
    Returns
    -------
    folds : list of lists
        Each element is a list of subject IDs for that fold
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(subject_ids)
    folds = np.array_split(shuffled, n_folds)
    return [list(fold) for fold in folds]

