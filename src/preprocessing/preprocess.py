import os
import librosa
import noisereduce as nr
import scipy.signal as signal
import numpy as np

def preprocess_single_audio(file_path):
    # 1. Load audio (sr=None preserves the original sample rate)
    y, sr = librosa.load(file_path, sr=None)

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


def get_audio_files(target_path):

    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    
    if os.path.isfile(target_path):
        if any(target_path.lower().endswith(ext) for ext in valid_extensions):
            yield target_path
            
    elif os.path.isdir(target_path):
        for root, dirs, files in os.walk(target_path):
            files.sort() 
            
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    yield os.path.join(root, file)
    else:
        print(f"Path not found: {target_path}")


def main_pipeline(target_path):

    for file_path in get_audio_files(target_path):
        try:
            print(f"Processing: {file_path}")
            
            y_clean, sr = preprocess_single_audio(file_path)
            

        except Exception as e:
            print(f"Failed to process {file_path}. Error: {e}")
