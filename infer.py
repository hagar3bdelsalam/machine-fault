import sys
import os
import glob
import re


from src.train.inference_utils import (
    load_model,
    get_mfcc_extractor,
    predict_single_audio,
)


def get_sorted_audio_files(data_dir):
    """
    Get all audio files from directory, sorted numerically.
    Supports: .wav, .mp3, .flac, .ogg
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Search for all supported audio formats
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(data_dir, ext)))
        # Also check uppercase extensions
        audio_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
    
    # Sort numerically by extracting the numeric part of the filename
    def extract_number(filename):
        # Extract just the filename without path
        base_filename = os.path.basename(filename)
        # Match numeric part in the filename
        match = re.search(r'(\d+)', base_filename)
        if match:
            return int(match.group(1))
        return float('inf')
    
    audio_files.sort(key=extract_number)
    
    if not audio_files:
        raise ValueError(f"No audio files found in {data_dir}")
    
    return audio_files


def infer(data_dir, model_path='src/model/model_epoch_75.pkl', output_dir='./results'):
    """
    Run inference on all audio files in the data directory.
    
    Args:
        data_dir: Directory containing audio files to process
        model_path: Path to the model weights file
        output_dir: Directory to save results to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, device = load_model(model_path)
    
    # Initialize feature extractor
    mfcc_extractor = get_mfcc_extractor()
    
    # Get sorted list of audio files
    audio_files = get_sorted_audio_files(data_dir)
    
    # Lists to store results and times
    predictions = []
    execution_times = []
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files, 1):
        try:
            # Run inference using generic function
            pred, elapsed = predict_single_audio(
                audio_file, model, mfcc_extractor, device,
                target_length=250, return_probs=False
            )
            predictions.append(str(pred))
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}", file=sys.stderr)
            # Default to class 0 on error
            predictions.append('0')
            elapsed = 0.0
        
        execution_times.append(f"{elapsed:.3f}")
    
    # Write results to file
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    # Write times to file
    times_file = os.path.join(output_dir, 'time.txt')
    with open(times_file, 'w') as f:
        for exec_time in execution_times:
            f.write(exec_time + '\n')
    
    print(f"Results saved to {results_file}", file=sys.stderr)
    print(f"Times saved to {times_file}", file=sys.stderr)
    print(f"Processing complete", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python infer.py <data_directory> [model_path] [output_directory]", file=sys.stderr)
        sys.exit(1)
    
    data_directory = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else 'src/model/model_epoch_75.pkl'
    output_directory = sys.argv[3] if len(sys.argv) > 3 else 'results/'
    
    infer(data_directory, model_file, output_directory)
