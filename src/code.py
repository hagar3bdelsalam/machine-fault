import os
import torch
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Import custom modules from your project architecture
from utils import load_features_and_labels, inspect_pkl
from dataloader.dataset import AudioDataset
from model.model import AudioClassifier
from train.train import Trainer
from train.test import ModelEvaluator

# --- Configuration ---
PROCESSED_DATA_PATH = './processed_data'
CHECKPOINT_DIR = './checkpoints'
BATCH_SIZE = 32
NUM_EPOCHS = 75

def main():
    print("="*50)
    print("1. Setup and Device Initialization")
    print("="*50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # --- Data Loading ---
    print("\n" + "="*50)
    print("2. Loading Processed Data")
    print("="*50)
    
    train_path = os.path.join(PROCESSED_DATA_PATH, 'train/train_data.pkl')
    test_path = os.path.join(PROCESSED_DATA_PATH, 'test/test_data.pkl')
    val_path = os.path.join(PROCESSED_DATA_PATH, 'validation/val_data.pkl')

    # Inspect the training data structure
    print("Inspecting training data pickle:")
    inspect_pkl(train_path)

    train_features, train_labels = load_features_and_labels(train_path)
    test_features, test_labels = load_features_and_labels(test_path)
    val_features, val_labels = load_features_and_labels(val_path)

    # --- Label Encoding ---
    print("\n" + "="*50)
    print("3. Label Encoding")
    print("="*50)
    label_encoder = LabelEncoder()
    all_labels = val_labels + test_labels + train_labels
    label_encoder.fit(all_labels)

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    train_labels_enc = label_encoder.transform(train_labels)
    test_labels_enc = label_encoder.transform(test_labels)
    val_labels_enc = label_encoder.transform(val_labels)

    # --- Datasets and DataLoaders ---
    print("\n" + "="*50)
    print("4. Creating DataLoaders")
    print("="*50)
    train_dataset = AudioDataset(train_features, train_labels_enc)
    test_dataset = AudioDataset(test_features, test_labels_enc)
    val_dataset = AudioDataset(val_features, val_labels_enc)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --- Model Initialization ---
    print("\n" + "="*50)
    print("5. Model Initialization")
    print("="*50)
    model = AudioClassifier(num_classes=num_classes)
    
    trainer = Trainer(model, train_loader, val_loader, device=device)
    trainer.count_parameters()

    # --- Training ---
    print("\n" + "="*50)
    print("6. Training Phase")
    print("="*50)
    
    # Fit the model (saves checkpoints automatically as .pkl)
    history = trainer.fit(num_epochs=NUM_EPOCHS, save_dir=CHECKPOINT_DIR)
    print(f"\nBest Validation Accuracy: {trainer.best_val_acc:.2f}%")

    # --- Testing & Evaluation ---
    print("\n" + "="*50)
    print("7. Testing & Evaluation Phase")
    print("="*50)
    
    evaluator = ModelEvaluator(
        model=trainer.model, # Use the model with the best loaded weights
        test_loader=test_loader, 
        device=device, 
        class_names=label_encoder.classes_
    )

    test_loss, test_accuracy, actual_labels, predicted_labels, timing_stats = evaluator.evaluate()

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%\n")

    evaluator.print_report(actual_labels, predicted_labels, timing_stats)
    
    # Save the confusion matrix plot
    plot_path = os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(actual_labels, predicted_labels, save_path=plot_path)

if __name__ == "__main__":
    main()