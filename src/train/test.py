import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, model, test_loader, device='cpu', class_names=None):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.class_names = class_names
        self.criterion = torch.nn.CrossEntropyLoss()

    def _sync_gpu(self):
        """Forces Python to wait for GPU operations to finish before taking a timestamp."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def evaluate(self):
        """Runs the test set and calculates loss, accuracy, and detailed inference times."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Timing metrics
        worst_batch_time = 0.0
        worst_sample_time = 0.0
        total_inference_time = 0.0

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                batch_size = features.size(0)

                # Start timer
                self._sync_gpu()
                start_time = time.perf_counter()

                # Forward pass
                outputs = self.model(features)

                # Stop timer
                self._sync_gpu()
                end_time = time.perf_counter()

                # Calculate times
                batch_time = end_time - start_time
                sample_time = batch_time / batch_size
                
                total_inference_time += batch_time
                worst_batch_time = max(worst_batch_time, batch_time)
                worst_sample_time = max(worst_sample_time, sample_time)

                # Calculate loss and metrics
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * batch_size

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        total_samples = len(self.test_loader.dataset)
        avg_loss = total_loss / total_samples
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = 100 * np.mean(all_preds == all_labels)
        
        avg_sample_time = total_inference_time / total_samples

        timing_stats = {
            'worst_batch_time_ms': worst_batch_time * 1000,
            'worst_sample_time_ms': worst_sample_time * 1000,
            'avg_sample_time_ms': avg_sample_time * 1000,
            'total_time_sec': total_inference_time
        }

        return avg_loss, accuracy, all_labels, all_preds, timing_stats

    def print_report(self, labels, preds, timing_stats):
        """Prints classification report and timing metrics."""
        print(f"{'='*50}\nInference Timing Analysis\n{'='*50}")
        print(f"Total Inference Time:  {timing_stats['total_time_sec']:.4f} sec")
        print(f"Average Time/Sample:   {timing_stats['avg_sample_time_ms']:.2f} ms")
        print(f"Worst Batch Time:      {timing_stats['worst_batch_time_ms']:.2f} ms")
        print(f"Worst Time/Sample:     {timing_stats['worst_sample_time_ms']:.2f} ms")
        print(f"\n{'='*50}\nClassification Report\n{'='*50}")
        
        target_names = [str(c) for c in self.class_names] if self.class_names is not None else None
        print(classification_report(labels, preds, target_names=target_names, zero_division=0))

    def plot_confusion_matrix(self, labels, preds, save_path=None):
        """Plots a heatmap of the confusion matrix."""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        plt.show()