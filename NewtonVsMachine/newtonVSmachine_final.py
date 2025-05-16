import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

inputf = pd.read_csv("train_X.csv").values
target = pd.read_csv("train_Y.csv").values

INPUT_DIM = inputf.shape[1] 
OUTPUT_DIM = target.shape[1]

#DEFINE THE MODEL
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape= INPUT_DIM),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(OUTPUT_DIM),
])

# Compile the model
model.compile(
    optimizer="adam",
    loss='mean_absolute_error',
    metrics=['mae']
)

# Train the model		
history = model.fit(
    inputf, target,
    validation_split=0.2,  # Use 10% of the data for validation
    epochs=100, 
    batch_size=5000,
    verbose=1
)

# Evaluate the model
loss, mae = model.evaluate(inputf, target, verbose=1)
print(f"Final MAE: {mae}")


# Plot Mean Absolute Error (MAE) vs. Epoch
plt.figure(figsize=(10, 8))
plt.plot(history.history['mae'], label='Training MAE', color='blue')
plt.plot(history.history['val_mae'], label='Validation MAE', color='orange', linestyle='--')

# Add labels and title
plt.title('Mean Absolute Error (MAE) vs Epoch', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.savefig("100000epochs.png")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Apply Savitzky-Golay filter to smooth the curves
window_size = 51  # Choose an odd number (e.g., 51) for the smoothing window
poly_order = 3    # Polynomial order for smoothing

# Smooth training and validation MAE
smooth_train_mae = savgol_filter(history.history['mae'], window_size, poly_order)
smooth_val_mae = savgol_filter(history.history['val_mae'], window_size, poly_order)

# Plot smoothed MAE vs. Epochs
plt.figure(figsize=(10, 6))
plt.plot(smooth_train_mae, label='Smoothed Training MAE', color='blue')
plt.plot(smooth_val_mae, label='Smoothed Validation MAE', color='orange')
plt.axhline(y=0.1, color='red', linestyle='--', label='MAE ≤ 0.1 Threshold')
plt.title("Smoothed MAE vs. Epochs (Log Scale)")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.legend()
plt.grid()
plt.savefig("smoothed_mae_vs_epochs_logscale.png", dpi=300, bbox_inches='tight')
plt.show()
