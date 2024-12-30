import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm  # For progress bars

# ===========================
# Configuration Parameters
# ===========================

# Paths to directories
data_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\processed_data'
train_output_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\mfcc_features_train'
val_output_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\mfcc_features_val'
label_encoder_path = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\label_encoder.pkl'

# Processing parameters
n_mfcc = 40           # Number of MFCC features to extract
test_size = 0.2       # 80% training, 20% validation
random_state = 42     # For reproducibility
sr = 16000            # Sampling rate

# ===========================
# Create Output Directories
# ===========================

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# ===========================
# Function Definitions
# ===========================

def extract_mfcc(audio_path, n_mfcc=40, sr=16000):
    """
    Extracts MFCC features from an audio file.

    Parameters:
    - audio_path (str): Path to the audio file.
    - n_mfcc (int): Number of MFCC features to extract.
    - sr (int): Sampling rate.

    Returns:
    - np.ndarray: MFCC feature matrix (time, features).
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.normalize(mfcc)  # Normalize MFCC
    return mfcc.T  # Transpose to shape (time, features)

def save_mfcc(mfcc, output_path):
    """
    Saves the MFCC feature matrix to a .npy file.

    Parameters:
    - mfcc (np.ndarray): MFCC feature matrix.
    - output_path (str): Path to save the .npy file.
    """
    np.save(output_path, mfcc)

# ===========================
# Initialize Label List
# ===========================

labels = []

# ===========================
# Processing Pipeline
# ===========================

print("Starting MFCC feature extraction and saving...")

# List all speakers
speakers = [speaker for speaker in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, speaker))]

# Initialize list to collect all speaker names for label encoding
all_speakers = []

for speaker in tqdm(speakers, desc="Processing Speakers"):
    speaker_dir = os.path.join(data_dir, speaker)
    
    # List all valid audio files for the current speaker
    audio_files = [f for f in os.listdir(speaker_dir) if f.lower().endswith(('.wav', '.flac', '.mp3', '.m4a'))]
    
    if not audio_files:
        print(f"  No valid audio files found for speaker '{speaker}'. Skipping.")
        continue
    
    # Split audio files into training and validation sets
    train_files, val_files = train_test_split(
        audio_files, test_size=test_size, random_state=random_state
    )
    
    # Define output directories for current speaker
    train_speaker_dir = os.path.join(train_output_dir, speaker)
    val_speaker_dir = os.path.join(val_output_dir, speaker)
    os.makedirs(train_speaker_dir, exist_ok=True)
    os.makedirs(val_speaker_dir, exist_ok=True)
    
    # Process training files
    for audio_file in tqdm(train_files, desc=f"  Processing Training Files for {speaker}", leave=False):
        audio_path = os.path.join(speaker_dir, audio_file)
        try:
            mfcc = extract_mfcc(audio_path, n_mfcc=n_mfcc, sr=sr)
            
            # Define output file name (e.g., audio1_chunk0.npy)
            base_name = os.path.splitext(audio_file)[0]
            output_filename = f"{base_name}.npy"
            output_path = os.path.join(train_speaker_dir, output_filename)
            
            # Save MFCC
            save_mfcc(mfcc, output_path)
            
            # Append label
            labels.append(speaker)
            all_speakers.append(speaker)
            
        except Exception as e:
            print(f"    Error processing '{audio_file}': {e}")
            continue
    
    # Process validation files
    for audio_file in tqdm(val_files, desc=f"  Processing Validation Files for {speaker}", leave=False):
        audio_path = os.path.join(speaker_dir, audio_file)
        try:
            mfcc = extract_mfcc(audio_path, n_mfcc=n_mfcc, sr=sr)
            
            # Define output file name
            base_name = os.path.splitext(audio_file)[0]
            output_filename = f"{base_name}.npy"
            output_path = os.path.join(val_speaker_dir, output_filename)
            
            # Save MFCC
            save_mfcc(mfcc, output_path)
            
            # Append label
            labels.append(speaker)
            all_speakers.append(speaker)
            
        except Exception as e:
            print(f"    Error processing '{audio_file}': {e}")
            continue

print("\nCompleted MFCC extraction and saving.")

# ===========================
# Label Encoding
# ===========================

# Get unique speakers
unique_speakers = sorted(list(set(all_speakers)))

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(unique_speakers)

# Save the label encoder
with open(label_encoder_path, 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
    print(f"Label encoder saved to {label_encoder_path}")

print(f"Number of classes (speakers): {len(label_encoder.classes_)}")

print("\nMFCC feature extraction and saving process is complete!")
