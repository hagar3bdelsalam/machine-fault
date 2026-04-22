import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import KFold


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def cross_validate(model_class, dataset, num_folds=10, num_epochs=50, device='cpu', batch_size=32):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{num_folds}")
        print(f"{'='*50}")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = 0
        best_model_state = None

        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:>3} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'model_state': best_model_state
        })

        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    return fold_results


def get_cross_validation_stats(fold_results):
    accuracies = [r['best_val_acc'] for r in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    return mean_acc, std_acc


def train_with_cross_validation(model_class, features, labels, target_size=(40, 40), num_folds=10, num_epochs=50, device='cpu', batch_size=32):
    from dataloader.dataset import AudioDataset
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    labels_enc = label_encoder.fit_transform(labels)

    dataset = AudioDataset(features, labels_enc, target_size=target_size)

    print(f"Running {num_folds}-fold cross validation...")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    fold_results = cross_validate(model_class, dataset, num_folds=num_folds, num_epochs=num_epochs, device=device, batch_size=batch_size)

    mean_acc, std_acc = get_cross_validation_stats(fold_results)

    print(f"\n{'='*50}")
    print(f"Cross Validation Results")
    print(f"{'='*50}")
    for result in fold_results:
        print(f"Fold {result['fold']}: {result['best_val_acc']:.2f}%")
    print(f"\nMean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")

    best_fold = max(fold_results, key=lambda x: x['best_val_acc'])
    print(f"\nBest Fold: {best_fold['fold']} with {best_fold['best_val_acc']:.2f}%")

    return fold_results, mean_acc, std_acc, label_encoder