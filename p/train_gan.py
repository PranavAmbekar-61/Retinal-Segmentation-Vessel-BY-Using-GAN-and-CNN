import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_generator():
    """Builds a simple autoencoder/generator model to act as our 'GAN'."""
    inputs = layers.Input(shape=(256, 256, 1))
    
    # Downsample
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    
    # Bottleneck
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    
    # Upsample
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    return models.Model(inputs, outputs)

def train_mock_gan():
    print("Building mock GAN model...")
    model = build_generator()
    model.compile(optimizer='adam', loss='mse')
    
    print("Generating dummy training data (black squares with white dots)...")
    X_train = np.zeros((10, 256, 256, 1), dtype=np.float32)
    Y_train = np.zeros((10, 256, 256, 1), dtype=np.float32)
    
    # Train for a few epochs
    print("Training model...")
    model.fit(X_train, Y_train, epochs=2, batch_size=2, verbose=1)
    
    # Save the model
    model_path = 'gan_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}!")

if __name__ == '__main__':
    train_mock_gan()
