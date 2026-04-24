import os
import librosa
import noisereduce as nr
import scipy.signal as signal
import numpy as np
import time
import torch

from src.features_extraction.MFCC import MfccExtractor
from src.model.model import AudioClassifier


DEFAULT_MODEL_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 2048,
    "n_mels": 40,
    "hop_length": 2048,
    "n_mfcc": 40,
    "target_length": 250,
}


def load_model(model_path, device=None):
    """
    Load the audio classifier model.
    
    Args:
        model_path: Path to the model weights file (.pkl)
        device: torch device (defaults to GPU if available, else CPU)
    
    Returns:
        model: AudioClassifier model in eval mode on specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.to(device)
    
    return model, device


def get_mfcc_extractor(sample_rate=None, n_fft=None, n_mels=None, 
                        hop_length=None, n_mfcc=None):
    """
    Create and return an MFCC feature extractor.
    
    Args:
        sample_rate, n_fft, n_mels, hop_length, n_mfcc: Feature extraction parameters.
                                                        Uses defaults if not provided.
    
    Returns:
        MfccExtractor: Configured feature extractor
    """
    config = DEFAULT_MODEL_CONFIG.copy()
    
    if sample_rate is not None:
        config["sample_rate"] = sample_rate
    if n_fft is not None:
        config["n_fft"] = n_fft
    if n_mels is not None:
        config["n_mels"] = n_mels
    if hop_length is not None:
        config["hop_length"] = hop_length
    if n_mfcc is not None:
        config["n_mfcc"] = n_mfcc
    
    return MfccExtractor(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        n_mels=config["n_mels"],
        hop_length=config["hop_length"],
        n_mfcc=config["n_mfcc"],
    )


def pad_or_truncate_features(features, target_length=250):
    """
    Pad or truncate MFCC features to a fixed length.
    
    Args:
        features: numpy array of shape (time_steps, n_mfcc)
        target_length: desired output time_steps dimension
    
    Returns:
        numpy array of shape (target_length, n_mfcc)
    """
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    time_steps = features.shape[0]
    
    if time_steps >= target_length:
        # Truncate
        return features[:target_length, :]
    else:
        # Pad with zeros
        padded = np.zeros((target_length, features.shape[1]))
        padded[:time_steps, :] = features
        return padded


def preprocess_single_audio(y, sr):

    # 2. Trim silence (top_db controls the threshold for silence, 40 is a good default)
    y_trimmed, _ = librosa.effects.trim(y, top_db=40)

    # 3. Apply spectral gating to remove ambient hiss
    y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr)

    # 4. Apply a Bandpass filter (20 Hz to 20 kHz)
    lowcut = 20.0
    nyquist = 0.5 * sr
    
    # CRITICAL: Prevent SciPy ValueError by keeping highcut below Nyquist
    highcut = min(20000.0, nyquist - 1.0) 

    if lowcut < highcut:
        low = lowcut / nyquist
        high = highcut / nyquist
        # 4th order butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')
        y_filtered = signal.lfilter(b, a, y_denoised)
    else:
        # Fallback in case the audio sample rate is extremely low
        y_filtered = y_denoised

    # 5. Normalize the volume
    y_normalized = librosa.util.normalize(y_filtered)

    return y_normalized, sr


def predict_single_audio(audio_file_path, model, mfcc_extractor, device,
                          target_length=250, return_probs=False):
    """
    Run inference on a single audio file.
    Timing excludes I/O operations (file reading).
    
    Args:
        audio_file_path: Path to the audio file
        model: AudioClassifier model
        mfcc_extractor: MfccExtractor instance
        device: torch device
        target_length: target length for feature padding/truncation
        return_probs: If True, return prediction probabilities
    
    Returns:
        prediction: Predicted class index
        probs: Softmax probabilities for all classes (if return_probs=True)
        elapsed_time: Time taken for feature extraction and inference (excludes I/O)
    """
    # Load audio file (I/O not timed)
    y, sr = librosa.load(audio_file_path, sr=None)

    # Start timer after file I/O
    start_time = time.time()
    
    y_clean, _sr = preprocess_single_audio(y, sr)
    # Extract features
    mfcc_features = mfcc_extractor.process(y_clean)
    
    # Pad/truncate
    mfcc_features = pad_or_truncate_features(mfcc_features, target_length=target_length)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).unsqueeze(0)
    features_tensor = features_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(features_tensor)
        prediction = int(torch.argmax(output, dim=1).item())
        # calculate probabilities if requested (for app.py display)
        if return_probs:
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        else:
            probs = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    
    if return_probs:
        return prediction, probs, elapsed_time
    else:
        return prediction, elapsed_time
