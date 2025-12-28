"""
CNN model for sequence classification using TensorFlow/Keras.
"""

from typing import Optional, Tuple, List
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    1D CNN Classifier using TensorFlow/Keras.
    Compatible with scikit-learn pipeline.
    """
    
    def __init__(
        self,
        input_shape: Optional[Tuple[int, int]] = None,
        n_classes: int = 3,
        n_filters: int = 32,
        kernel_size: int = 3,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: int = 1
    ):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.verbose = verbose
        
        self.model = None
        self.history = None
        self.le = LabelEncoder()

    def _build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(filters=self.n_filters, kernel_size=self.kernel_size, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        # Encode labels if they are not integers
        if y.dtype == 'object' or isinstance(y[0], str):
            y = self.le.fit_transform(y)
        
        # Reshape X for CNN if needed: (samples, features) -> (samples, features, 1)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
        self.input_shape = (X.shape[1], X.shape[2])
        self.model = self._build_model(self.input_shape)
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        
        probs = self.model.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        if hasattr(self.le, 'classes_'):
            return self.le.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X, verbose=0)
