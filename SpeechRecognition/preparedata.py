import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Set directories
data_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\mfcc_features'

# Initialize lists for data and labels
X = []  # Features (MFCCs)
y = []  # Labels (Speakers)

# Iterate through each speaker's folder
for speaker in os.listdir(data_dir):
    speaker_dir = os.path.join(data_dir, speaker)
    if os.path.isdir(speaker_dir):
        
        # Iterate through each .npy file (MFCC)
        for audio_file in os.listdir(speaker_dir):
            if audio_file.endswith('.npy'):
                audio_path = os.path.join(speaker_dir, audio_file)
                
                # Load the MFCC feature
                mfcc = np.load(audio_path)
                
                # Optionally, pad the MFCC to have consistent time steps across samples
                mfcc_padded = pad_sequences([mfcc], padding='post', dtype='float32', maxlen=300)  # Adjust maxlen if needed
                mfcc_padded = mfcc_padded[0]  # Remove extra list dimension
                
                # Append the MFCC and label (speaker)
                X.append(mfcc_padded)
                y.append(speaker)

# Convert X and y into numpy arrays
X = np.array(X)
y = np.array(y)

# Label encoding for the speakers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 2: Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Optionally, reshape X to match model input (if needed)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # Adding channel dimension if needed
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)  # Adding channel dimension if needed

# Print the data shapes
print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
