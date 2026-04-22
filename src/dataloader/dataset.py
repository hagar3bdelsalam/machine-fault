import torch
from torch.utils.data import Dataset
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, features, labels, max_frames=250, n_mfcc=40):
        self.features = features
        self.labels = labels
        self.max_frames = max_frames
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        # 1. Standardize the Time Dimension (frames)
        if feature.shape[0] > self.max_frames:
            # Truncate if longer than max_frames
            feature = feature[:self.max_frames, :]
        elif feature.shape[0] < self.max_frames:
            # Zero-pad if shorter than max_frames
            pad_width = self.max_frames - feature.shape[0]
            feature = np.pad(feature, ((0, pad_width), (0, 0)), mode='constant')

        # 2. Standardize the MFCC Dimension (just to be safe)
        if feature.shape[1] > self.n_mfcc:
            feature = feature[:, :self.n_mfcc]
        elif feature.shape[1] < self.n_mfcc:
            pad_width = self.n_mfcc - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')

        # 3. Add channel dimension for Conv2d: shape becomes (1, max_frames, n_mfcc)
        feature = np.expand_dims(feature, axis=0)

        return torch.FloatTensor(feature), torch.LongTensor([self.labels[idx]])[0]
    