import numpy as np
import cv2
import os
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import time

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f     = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f     = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)

def combined_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

CUSTOM_OBJECTS = {
    'combined_loss': combined_loss,
    'bce_dice_loss': bce_dice_loss,
    'dice_loss':     dice_loss,
    'dice_coef':     dice_coef,
}

print("Loading U-Net model...")
unet_model = load_model('model.h5', custom_objects=CUSTOM_OBJECTS)
print(f"U-Net loaded. Input shape: {unet_model.input_shape}")

print("Loading mock GAN model...")
if os.path.exists('gan_model.h5'):
    gan_model = load_model('gan_model.h5', compile=False)
    print(f"GAN loaded. Input shape: {gan_model.input_shape}")
else:
    print("gan_model.h5 not found!")
    gan_model = None

# Create a dummy image mimicking a preprocessed green channel
print("Generating dummy test image (256x256x1)...")
dummy_img = np.random.rand(1, 256, 256, 1).astype(np.float32)

print("Running U-Net prediction...")
t0 = time.time()
unet_pred = unet_model.predict(dummy_img, verbose=0)
t1 = time.time()
print(f"U-Net Prediction successful. Time: {(t1-t0)*1000:.2f} ms")
print(f"U-Net Output shape: {unet_pred.shape}, Min: {unet_pred.min():.4f}, Max: {unet_pred.max():.4f}")

if gan_model:
    print("Running GAN prediction...")
    t0 = time.time()
    gan_pred = gan_model.predict(dummy_img, verbose=0)
    t1 = time.time()
    print(f"GAN Prediction successful. Time: {(t1-t0)*1000:.2f} ms")
    print(f"GAN Output shape: {gan_pred.shape}, Min: {gan_pred.min():.4f}, Max: {gan_pred.max():.4f}")

print("\nAll ML models executed successfully! No crashes.")
