# finetune_model.py

import os
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, regularizers, optimizers, constraints
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
import numpy as np
import keras

# Suppress TensorFlow warnings for cleaner output (optional)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Ensure correct path for model imports
# Update this path if your project structure changes
sys.path.append(
    r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\VGG-Speaker-Recognition\src'
)

# Import necessary components from model.py
from model import vggvox_resnet2d_icassp

# Custom data generator for MFCC files
class MFCCDataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=32, shuffle=True, target_size=(250, 40)):
        """
        Initializes the data generator.

        Parameters:
        - data_dir (str): Directory containing MFCC .npy files organized by speaker.
        - batch_size (int): Number of samples per batch.
        - shuffle (bool): Whether to shuffle data after each epoch.
        - target_size (tuple): Desired shape of the MFCC features (time_steps, n_mfcc).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size  # (time_steps, n_mfcc)
        self.classes = sorted(os.listdir(data_dir))  # List of speaker folders
        self.num_classes = len(self.classes)
        self.class_indices = {cls: idx for idx, cls in enumerate(self.classes)}
        self.file_paths = []
        self.labels = []

        # Collect file paths and labels
        for speaker in self.classes:
            speaker_dir = os.path.join(data_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for audio_file in os.listdir(speaker_dir):
                if audio_file.endswith('.npy'):
                    self.file_paths.append(os.path.join(speaker_dir, audio_file))
                    self.labels.append(self.class_indices[speaker])

        self.labels = np.array(self.labels)
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        """Shuffles data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Generates one batch of data.

        Parameters:
        - index (int): Index of the batch.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Batch of MFCC features and corresponding one-hot labels.
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_file_paths = [self.file_paths[i] for i in batch_indices]
        batch_labels = self.labels[batch_indices]

        batch_X = []
        for file_path in batch_file_paths:
            try:
                mfcc = np.load(file_path)  # Load MFCC feature

                # Pad or truncate MFCC to match target_size
                if mfcc.shape[0] > self.target_size[0]:
                    mfcc = mfcc[:self.target_size[0], :]
                else:
                    pad_width = self.target_size[0] - mfcc.shape[0]
                    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

                if mfcc.shape[1] > self.target_size[1]:
                    mfcc = mfcc[:, :self.target_size[1]]
                else:
                    pad_width = self.target_size[1] - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

                # Add channel dimension
                mfcc = np.expand_dims(mfcc, axis=-1)  # Shape: (time_steps, n_mfcc, 1)

                batch_X.append(mfcc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue  # Skip the sample

        batch_X = np.array(batch_X, dtype=np.float32)

        # One-hot encode labels
        batch_labels = keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        return batch_X, batch_labels

# Configuration class to hold arguments
class Config:
    def __init__(self, args_dict):
        for key, value in args_dict.items():
            setattr(self, key, value)

# Define your model parameters
args_dict = {
    'net': 'resnet34s',         # Model architecture
    'loss': 'softmax',          # Loss function
    'vlad_cluster': 64,
    'ghost_cluster': 0,
    'bottleneck_dim': 512,
    'aggregation_mode': 'vlad', # Aggregation mode
    'optimizer': 'adam',        # Optimizer type
    'num_classes': 8,           # Number of speaker classes
}
args = Config(args_dict)

# Define target size based on MFCC extraction
target_size = (250, 40)  # (time_steps, n_mfcc)

# Initialize the model
print(f"Model Arguments: {vars(args)}")
model = vggvox_resnet2d_icassp(
    input_dim=(target_size[0], target_size[1], 1),  # Adjusted to match MFCC shape
    num_class=args.num_classes,
    mode='train',
    args=args
)

# Print model summary to verify architecture
print("\nModel Summary:")
model.summary()

# Save initial weights for comparison
initial_weights = {}
for layer in model.layers:
    initial_weights[layer.name] = layer.get_weights()

# Load the pretrained weights
pretrained_weights_path = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\models\weights.h5'

# Verify the existence of weights.h5
print(f"\nPretrained weights path: {pretrained_weights_path}")
if os.path.exists(pretrained_weights_path):
    print("Pretrained weights file found.")
else:
    print("Pretrained weights file NOT found. Please check the path and filename.")
    sys.exit(1)

# Attempt to load the weights
try:
    print(f"\nAttempting to load weights from: {pretrained_weights_path}")
    model.load_weights(pretrained_weights_path, by_name=True, skip_mismatch=False)
    print("Pretrained weights loaded successfully.")
except ValueError as ve:
    print(f"ValueError: {ve}")
    print("Ensure that the weights file matches the model architecture.")
    sys.exit(1)
except FileNotFoundError as fnf_error:
    print(f"FileNotFoundError: {fnf_error}")
    print("The weights file was not found. Please check the path and ensure the file exists.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading weights: {e}")
    sys.exit(1)

# Check which layers have updated weights
print("\nChecking which layers have updated weights:")
for layer in model.layers:
    layer_name = layer.name
    initial_layer_weights = initial_weights.get(layer_name)
    current_layer_weights = layer.get_weights()
    if initial_layer_weights:
        if len(initial_layer_weights) != len(current_layer_weights):
            print(f"Layer '{layer_name}' weights were updated.")
        else:
            weight_updated = False
            for w1, w2 in zip(initial_layer_weights, current_layer_weights):
                if not np.array_equal(w1, w2):
                    weight_updated = True
                    break
            if weight_updated:
                print(f"Layer '{layer_name}' weights were updated.")
            else:
                print(f"Layer '{layer_name}' weights were NOT updated.")
    else:
        print(f"Layer '{layer_name}' had no initial weights.")

# Freeze the earlier layers (keep the last few layers trainable)
# Adjust 'freeze_layers' based on your model's architecture
freeze_layers = 100  # Number of layers to freeze (adjust as needed)
for layer in model.layers[:freeze_layers]:
    layer.trainable = False
for layer in model.layers[freeze_layers:]:
    layer.trainable = True
print(f"\nFroze the first {freeze_layers} layers of the model.")

# Compile the model with a fine-tuning optimizer (use a smaller learning rate)
if args.optimizer.lower() == 'adam':
    opt = optimizers.Adam(learning_rate=1e-5)  # Reduced learning rate for fine-tuning
elif args.optimizer.lower() == 'sgd':
    opt = optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
else:
    raise ValueError("Unknown optimizer specified.")

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled with optimizer:", args.optimizer)

# Set up data generators for your training and validation datasets
train_data_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\mfcc_features_train'
val_data_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\mfcc_features_val'

# Verify data directories
if not os.path.isdir(train_data_dir):
    print(f"Training data directory not found: {train_data_dir}")
    sys.exit(1)
if not os.path.isdir(val_data_dir):
    print(f"Validation data directory not found: {val_data_dir}")
    sys.exit(1)

# Initialize data generators
train_generator = MFCCDataGenerator(train_data_dir, batch_size=32, shuffle=True, target_size=target_size)
val_generator = MFCCDataGenerator(val_data_dir, batch_size=32, shuffle=False, target_size=target_size)
print(f"\nTraining samples: {len(train_generator.file_paths)}, Validation samples: {len(val_generator.file_paths)}")

# Set up callbacks to save the best model and stop early if needed
callbacks = [
    ModelCheckpoint(
        filepath='finetuned_model_epoch{epoch:02d}_val_loss{val_loss:.2f}.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# Train the model
epochs = 50  # Adjust based on your needs
try:
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks
    )
    print("Model training complete.")
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)

# After training, save the final model
final_model_path = 'final_finetuned_model.keras'
try:
    model.save(final_model_path)
    print(f"\nFinal fine-tuned model saved to {final_model_path}")
except Exception as e:
    print(f"Error saving the final model: {e}")
