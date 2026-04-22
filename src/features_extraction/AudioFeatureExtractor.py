import numpy as np
from abc import ABC, abstractmethod

class AudioFeatureExtractor(ABC):
    """Abstract base class for all audio feature extraction."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    @abstractmethod
    def process(self, signal: np.ndarray) -> np.ndarray:
        """Processes a 1D audio signal and returns the extracted features."""
        pass
