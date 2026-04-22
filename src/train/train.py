import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import re

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_val_acc = 0
        self.best_model_state = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in self.train_loader:
            if isinstance(features, list):
                for feat, label in zip(features, labels):
                    feat = feat.unsqueeze(0).to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(feat)
                    loss = self.criterion(outputs, label.unsqueeze(0))
                    loss.backward()
                    self.optimizer.step()

                    # FIXED: Math scaling for single item
                    total_loss += loss.item() * 1 
                    _, predicted = torch.max(outputs, 1)
                    total += 1
                    correct += (predicted == label).sum().item()
            else:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # FIXED: Math scaling for batched items
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / total, 100 * correct / total

    def evaluate(self, loader=None):
        if loader is None:
            loader = self.val_loader
            
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                if isinstance(features, list):
                    for feat, label in zip(features, labels):
                        feat = feat.unsqueeze(0).to(self.device)
                        label = label.to(self.device)

                        outputs = self.model(feat)
                        loss = self.criterion(outputs, label.unsqueeze(0))

                        # FIXED: Math scaling for single item
                        total_loss += loss.item() * 1
                        _, predicted = torch.max(outputs, 1)
                        total += 1
                        correct += (predicted == label).sum().item()
                else:
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)

                    # FIXED: Math scaling for batched items
                    total_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        return total_loss / total, 100 * correct / total

    def fit(self, num_epochs=50, verbose=True, save_dir='/kaggle/working/checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                # FIXED: Deep copy to CPU
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # NEW: Save model every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), save_path)
                if verbose:
                    print(f"--> Saved checkpoint: {save_path}")

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch+1:>3} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return self.history

    def predict(self, loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features, _ in loader:
                if isinstance(features, list):
                    for feat in features:
                        feat = feat.unsqueeze(0).to(self.device)
                        outputs = self.model(feat)
                        _, predicted = torch.max(outputs, 1)
                        predictions.append(predicted.cpu().item())
                else:
                    features = features.to(self.device)
                    outputs = self.model(features)
                    _, predicted = torch.max(outputs, 1)
                    predictions.extend(predicted.cpu().numpy())
        return predictions


    def load_latest_checkpoint(self, checkpoints_dir='/kaggle/working/checkpoints'):
        import os
        import glob
        import re
        
        if not os.path.exists(checkpoints_dir):
            print("No checkpoint directory found. Starting fresh.")
            return 0

        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, 'model_epoch_*.pth'))
        if not checkpoint_files:
            print("No checkpoints found in directory. Starting fresh.")
            return 0

        latest_file = None
        max_epoch = -1
        
        for file in checkpoint_files:
            match = re.search(r'model_epoch_(\d+)\.pth', file)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_file = file

        if latest_file:
            # FIXED: Use self.model instead of model
            self.model.load_state_dict(torch.load(latest_file))
            print(f"Successfully resumed from {latest_file} (Epoch {max_epoch})")
            return max_epoch
        
        return 0