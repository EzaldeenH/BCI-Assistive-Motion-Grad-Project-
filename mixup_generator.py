import numpy as np
import tensorflow as tf
from tensorflow import keras
import re

class MixupGenerator(keras.utils.Sequence):
    """
    Data generator that applies Mixup augmentation to EEG data.
    Supports 'linear' and 'channel' mixup strategies from the mixEEG paper.
    Now supports 'Binary' channel mixup (Left vs Right hemisphere).
    """
    def __init__(self, X, y, batch_size=32, alpha=0.2, mixup_type='linear', shuffle=True, ch_names=None):
        """
        Initialization
        :param X: Input data (samples, 1, channels, timepoints)
        :param y: One-hot encoded labels (samples, n_classes)
        :param batch_size: Batch size
        :param alpha: Mixup interpolation coefficient (Beta distribution parameter)
        :param mixup_type: 'linear', 'channel', or None
        :param shuffle: Whether to shuffle data at the beginning of each epoch
        :param ch_names: List of channel names for binary mixup
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.mixup_type = mixup_type
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.n_channels = X.shape[2]  # Assuming shape (N, 1, Chans, Time)
        self.ch_names = ch_names
        
        # Pre-compute Left/Right indices if channel names are provided
        self.left_indices = []
        self.right_indices = []
        if self.ch_names is not None and self.mixup_type == 'channel':
            self._compute_hemisphere_indices()
            
        self.on_epoch_end()

    def _compute_hemisphere_indices(self):
        """
        Parses channel names to find Left (odd/z) and Right (even) indices.
        According to 10-20/10-05 system: Odd=Left, Even=Right, z=Center.
        We will assign Center (z) to Left group arbitrarily to ensure coverage, 
        or keep them as distinct groups. 
        Paper suggests 'Binary' split. We will split into {Left, Center} vs {Right}.
        """
        left = []
        right = []
        
        for idx, name in enumerate(self.ch_names):
            # Normalize name
            name = name.upper()
            
            # Check last character
            last_char = name[-1]
            
            if last_char.isdigit():
                val = int(last_char)
                if val % 2 != 0: # Odd -> Left
                    left.append(idx)
                else: # Even -> Right
                    right.append(idx)
            elif 'Z' in name: # Center (Fz, Cz, Pz, etc.)
                # Assign center to Left group for the binary split 
                # (or we could treat them as a third group, but binary implies 2)
                left.append(idx)
            else:
                # Fallback: Assign to left
                left.append(idx)
                
        self.left_indices = np.array(left)
        self.right_indices = np.array(right)
        print(f"MixupGenerator: Identified {len(self.left_indices)} Left+Center channels and {len(self.right_indices)} Right channels.")

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        if self.mixup_type == 'linear':
            return self._linear_mixup(X_batch, y_batch)
        elif self.mixup_type == 'channel':
            if len(self.left_indices) > 0 and len(self.right_indices) > 0:
                return self._channel_mixup_binary(X_batch, y_batch)
            else:
                return self._channel_mixup_random(X_batch, y_batch)
        else:
            return X_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _linear_mixup(self, X_batch, y_batch):
        """
        Applies Linear Mixup: x = lambda*x1 + (1-lambda)*x2
        """
        batch_size = len(X_batch)
        l = np.random.beta(self.alpha, self.alpha, batch_size)
        
        X_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)

        perm_indices = np.random.permutation(batch_size)
        X_batch_2 = X_batch[perm_indices]
        y_batch_2 = y_batch[perm_indices]

        X_mix = X_l * X_batch + (1 - X_l) * X_batch_2
        y_mix = y_l * y_batch + (1 - y_l) * y_batch_2

        return X_mix, y_mix

    def _channel_mixup_random(self, X_batch, y_batch):
        """
        Fallback: Random Channel Mixup (if no channel names provided)
        """
        batch_size = len(X_batch)
        perm_indices = np.random.permutation(batch_size)
        X_batch_2 = X_batch[perm_indices]
        y_batch_2 = y_batch[perm_indices]

        mask = np.random.randint(0, 2, size=(batch_size, 1, self.n_channels, 1)).astype(np.float32)
        
        kept_ratio = np.sum(mask, axis=2) / self.n_channels
        kept_ratio_y = kept_ratio.reshape(batch_size, 1)

        X_mix = mask * X_batch + (1 - mask) * X_batch_2
        y_mix = kept_ratio_y * y_batch + (1 - kept_ratio_y) * y_batch_2

        return X_mix, y_mix

    def _channel_mixup_binary(self, X_batch, y_batch):
        """
        Binary Channel Mixup: Swaps Hemisphere Regions.
        For each sample, we decide whether to take:
        1. (Left_A + Right_B) OR (Left_B + Right_A)
        """
        batch_size = len(X_batch)
        perm_indices = np.random.permutation(batch_size)
        X_batch_2 = X_batch[perm_indices]
        y_batch_2 = y_batch[perm_indices]

        # Initialize mask with zeros
        # Shape: (Batch, 1, Chans, 1)
        mask = np.zeros((batch_size, 1, self.n_channels, 1), dtype=np.float32)

        # For each sample in batch, decide which mix to perform
        # 0: Keep Left_A, Swap Right_B (take right from 2)
        # 1: Keep Right_A, Swap Left_B (take left from 2)
        decision = np.random.randint(0, 2, size=batch_size)
        
        for i in range(batch_size):
            if decision[i] == 0:
                # Keep Left indices (mask=1), Swap Right indices (mask=0)
                mask[i, 0, self.left_indices, 0] = 1.0
                mask[i, 0, self.right_indices, 0] = 0.0
            else:
                # Keep Right indices (mask=1), Swap Left indices (mask=0)
                mask[i, 0, self.right_indices, 0] = 1.0
                mask[i, 0, self.left_indices, 0] = 0.0

        # Mixing logic is same: X_mix = mask * X1 + (1-mask) * X2
        X_mix = mask * X_batch + (1 - mask) * X_batch_2
        
        # Label mixing: based on ratio of channels kept
        # Since groups might not be equal size (e.g. Center included in Left),
        # we calculate ratio dynamically.
        kept_ratio = np.sum(mask, axis=2) / self.n_channels
        kept_ratio_y = kept_ratio.reshape(batch_size, 1)
        
        y_mix = kept_ratio_y * y_batch + (1 - kept_ratio_y) * y_batch_2

        return X_mix, y_mix
