import numpy as np
from .AudioFeatureExtractor import AudioFeatureExtractor

class STFTFeatureExtractor(AudioFeatureExtractor):
    """Base class for features relying on Short-Time Fourier Transform (STFT)."""
    
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160):
        super().__init__(sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pre_emphasis_coeff = 0.97
        self.window = self._init_window()

    def _init_window(self):
        """Standard Hamming Window."""
        n = np.arange(self.n_fft)
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (self.n_fft - 1))

    def get_power_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Executes the core STFT pipeline: pre-emphasis, framing, windowing, and FFT."""
        # 1. Pre-emphasis
        pre_emphasized = np.append(signal[0], signal[1:] - self.pre_emphasis_coeff * signal[:-1])
        
        n_frames = 1 + (len(pre_emphasized) - self.n_fft) // self.hop_length
        if n_frames < 1:
            return np.empty((0, self.n_fft // 2 + 1))
            
        # 2. Fast framing via memory striding
        shape = (n_frames, self.n_fft)
        strides = (pre_emphasized.strides[0] * self.hop_length, pre_emphasized.strides[0])
        frames = np.lib.stride_tricks.as_strided(pre_emphasized, shape=shape, strides=strides)
        
        # 3. Apply window
        frames = frames * self.window
        
        # 4. Radix-2 FFT and Power Spectrum
        complex_spec = np.fft.rfft(frames, n=self.n_fft, axis=1)
        power_spec = np.abs(complex_spec) ** 2 / self.n_fft
        
        return power_spec