import os
import pickle


def get_all_audio_files_with_labels(base_path):
    """
    Scans the dataset directory and returns a list of (file_path, class_label) tuples.
    Expected structure: base_path/Machine X/machine_data/Normal or Abnormal
    """
    audio_files = []
    
    if os.path.isdir(base_path):
        for machine_name in sorted(os.listdir(base_path)):
            machine_path = os.path.join(base_path, machine_name)
            if os.path.isdir(machine_path):
                for subfolder in sorted(os.listdir(machine_path)):
                    subfolder_path = os.path.join(machine_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        for state_name in sorted(os.listdir(subfolder_path)):
                            state_path = os.path.join(subfolder_path, state_name)
                            if os.path.isdir(state_path):
                                label = f"{machine_name}_{state_name}"
                                for root, dirs, files in os.walk(state_path):
                                    for file in sorted(files):
                                        if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                                            audio_files.append((os.path.join(root, file), label))
    
    return audio_files

def load_features_and_labels(pkl_file_path):
    """Loads features and labels from a pickle file."""
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
        
    if isinstance(data, dict):
        features_list = data['features']
        raw_labels_list = data['labels']
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        features_list, raw_labels_list = data
    else:
        raise ValueError("Unrecognized data structure in the pickle file.")
        
    return features_list, raw_labels_list

def inspect_pkl(file_path):
    """Prints a summary of the contents within a pickle file for debugging."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        print(f"Number of features: {len(data.get('features', []))}")
        labels = data.get('labels', [])
        print(f"Number of labels: {len(labels)}")
        print(f"Unique labels: {set(labels)}")
    else:
        print(f"Data type: {type(data)}")
        print(f"Length: {len(data)}")

def process_and_save_files(file_list, mfcc_extractor, preprocess_func):
    """
    Process audio files: preprocess + MFCC extraction and return features.
    Pass preprocess_single_audio as preprocess_func to keep utils generic.
    """
    features_list = []
    labels_list = []
    
    for file_path, class_label in file_list:
        try:
            # Preprocess audio
            y_clean, sr = preprocess_func(file_path)
            
            # Extract MFCC features
            mfcc_features = mfcc_extractor.process(y_clean)
            
            features_list.append(mfcc_features)
            labels_list.append(class_label)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return features_list, labels_list