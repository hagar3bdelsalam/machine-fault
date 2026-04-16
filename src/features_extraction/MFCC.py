import numpy as np
from .LogMelSpectrogram import LogMelSpectrogram

class MfccExtractor(LogMelSpectrogram):
    """Extracts MFCCs by applying DCT to the Log Mel Spectrogram."""
    
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=40, hop_length=160, n_mfcc=13, f_min=0.0, f_max=None):
        super().__init__(sample_rate, n_fft, n_mels, hop_length, f_min, f_max)
        self.n_mfcc = n_mfcc
        self.dct_matrix = self._init_dct_matrix()

    def _init_dct_matrix(self):
        """Constructs the Discrete Cosine Transform (DCT-II) matrix."""
        n = np.arange(self.n_mels)
        k = np.arange(self.n_mfcc)[:, np.newaxis]
        
        dct_mat = np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * self.n_mels))
        dct_mat *= np.sqrt(2.0 / self.n_mels)
        dct_mat[0, :] *= np.sqrt(0.5) 
        
        return dct_mat

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Decorrelates the Log Mel features using DCT."""
        log_mels = super().process(signal)
        
        if log_mels.size == 0:
            return np.empty((0, self.n_mfcc))
            
        mfcc_frames = np.dot(log_mels, self.dct_matrix.T)
        return mfcc_frames