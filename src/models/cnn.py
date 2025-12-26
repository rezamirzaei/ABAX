"""
CNN model for sequence classification using PyTorch.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class CNNModule(nn.Module):
    """1D CNN PyTorch Module."""
    def __init__(self, input_channels: int, n_classes: int, n_filters: int = 32, kernel_size: int = 3):
        super(CNNModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)

        # Calculate dense input size dynamically
        # We'll do a dummy forward pass in __init__ or just use a linear layer that adapts?
        # For simplicity, we'll use a LazyLinear or just calculate it.
        # Since input length varies, we might need GlobalMaxPooling to handle variable lengths,
        # or assume fixed length. The previous TF model assumed fixed length input_shape.
        # Let's use AdaptiveMaxPool1d to handle variable lengths and simplify dimension calculation.
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(n_filters, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x shape: (batch, features, sequence_length) or (batch, sequence_length, features)?
        # PyTorch Conv1d expects (batch, channels, length)
        # Our input is usually (samples, features) -> we treat features as channels?
        # Or (samples, time_steps, features)?
        # In the previous TF code: X.reshape((X.shape[0], X.shape[1], 1)) -> (samples, features, 1)
        # So features were treated as time steps, and channels=1.

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    1D CNN Classifier using PyTorch.
    Compatible with scikit-learn pipeline.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_filters: int = 32,
        kernel_size: int = 3,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: int = 1,
        device: str = "cpu"
    ):
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.verbose = verbose
        self.device = device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu"

        # Try to use MPS (Metal Performance Shaders) on macOS if available
        if torch.backends.mps.is_available() and device == "cpu":
             self.device = "mps"

        self.model = None
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        self.le = LabelEncoder()

    def fit(self, X, y):
        # Encode labels
        if y.dtype == 'object' or isinstance(y[0], str):
            y = self.le.fit_transform(y)

        # Prepare data
        # X shape: (samples, features)
        # Reshape for PyTorch Conv1d: (samples, channels, length)
        # We treat features as length, channels=1
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Split validation
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)
        n_train = n_samples - n_val

        # Shuffle and split
        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize model
        self.model = CNNModule(
            input_channels=1,
            n_classes=self.n_classes,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss /= total
            train_acc = correct / total

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss /= total
            val_acc = correct / total

            # Update history
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}: '
                      f'Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        return self

    def predict(self, X):
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        preds = predicted.cpu().numpy()

        if hasattr(self.le, 'classes_'):
            return self.le.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

