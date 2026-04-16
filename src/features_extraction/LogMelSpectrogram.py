import numpy as np
from .STFT import STFTFeatureExtractor

class LogMelSpectrogram(STFTFeatureExtractor):
    """Extracts Log Mel Spectrogram using the STFT pipeline."""
    
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=40, hop_length=160, f_min=0.0, f_max=None):
        super().__init__(sample_rate, n_fft, hop_length)
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.min_log_val = 1e-10
        self.mel_filterbank = self._init_filterbank()

    @staticmethod
    def _hz_to_mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m):
        return 700.0 * (np.exp(m / 1127.0) - 1.0)

    def _init_filterbank(self):
        """Constructs the triangular Mel filterbank matrix."""
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        n_freqs = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, n_freqs))
        
        for m in range(1, self.n_mels + 1):
            left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
            
            for k in range(left, center):
                filterbank[m - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                filterbank[m - 1, k] = (right - k) / (right - center)
                
        return filterbank

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Applies Mel filterbank to the power spectrum."""
        power_spec = self.get_power_spectrogram(signal)
        
        # Guard clause for empty sequences
        if power_spec.size == 0:
            return np.empty((0, self.n_mels))
            
        mel_energy = np.dot(power_spec, self.mel_filterbank.T)
        log_mel_energy = np.log(np.maximum(mel_energy, self.min_log_val))
        
        return log_mel_energy
