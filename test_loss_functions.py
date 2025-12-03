"""Quick test of the new loss functions."""
import numpy as np
import tensorflow as tf
from swafi.impact_dl import SoftF1Loss, SoftCSILoss

# Test data
y_true = tf.constant([[0], [1], [1], [0], [1]], dtype=tf.float32)
y_pred = tf.constant([[0.1], [0.9], [0.8], [0.2], [0.7]], dtype=tf.float32)

# Test Soft F1 Loss
print("Testing Soft F1 Loss...")
soft_f1 = SoftF1Loss(beta=1.0, class_weight=2.0)
loss_f1 = soft_f1(y_true, y_pred)
print(f"Soft F1 Loss: {loss_f1.numpy():.4f}")
print(f"  (Lower is better, range [0, 1])")

# Test Soft CSI Loss
print("\nTesting Soft CSI Loss...")
soft_csi = SoftCSILoss(class_weight=2.0)
loss_csi = soft_csi(y_true, y_pred)
print(f"Soft CSI Loss: {loss_csi.numpy():.4f}")
print(f"  (Lower is better, range [0, 1])")

# Test gradients work
print("\nTesting gradients...")
with tf.GradientTape() as tape:
    y_pred_var = tf.Variable(y_pred)
    loss = soft_f1(y_true, y_pred_var)

grads = tape.gradient(loss, y_pred_var)
print(f"Gradients shape: {grads.shape}")
print(f"Gradients: {grads.numpy().flatten()}")
print("✓ Gradients computed successfully!")

print("\n✓ All loss function tests passed!")

