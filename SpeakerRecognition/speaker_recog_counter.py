import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
import joblib
from tqdm import tqdm
import soundfile as sf

# ----------------------------
# Define the Custom VladPooling Layer
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class VladPooling(layers.Layer):
    def __init__(self, k_centers=64, mode='vlad', **kwargs):
        super(VladPooling, self).__init__(**kwargs)
        self.k_centers = k_centers
        self.mode = mode

    def build(self, input_shape):
        # Initialize cluster centers
        self.cluster_centers = self.add_weight(
            shape=(self.k_centers, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='cluster_centers'
        )
        super(VladPooling, self).build(input_shape)

    def call(self, inputs):
        if self.mode == 'vlad':
            # Flatten spatial dimensions: (batch_size, H, W, C) -> (batch_size, H*W, C)
            batch_size = tf.shape(inputs)[0]
            H = tf.shape(inputs)[1]
            W = tf.shape(inputs)[2]
            C = tf.shape(inputs)[3]
            inputs_flat = tf.reshape(inputs, [batch_size, H * W, C])  # (batch_size, H*W, C)

            # Compute soft assignments: (batch_size, H*W, k_centers)
            assignments = tf.nn.softmax(tf.matmul(inputs_flat, self.cluster_centers, transpose_b=True))

            # Compute residuals: (batch_size, H*W, k_centers, C)
            residuals = tf.expand_dims(inputs_flat, 2) - tf.expand_dims(self.cluster_centers, 0)

            # Weighted residuals: (batch_size, H*W, k_centers, C)
            weighted_residuals = tf.expand_dims(assignments, -1) * residuals

            # Aggregate: (batch_size, k_centers, C)
            vlad = tf.reduce_sum(weighted_residuals, axis=1)

            # Flatten: (batch_size, k_centers * C)
            vlad = tf.reshape(vlad, [batch_size, -1])

            # Normalize
            vlad = tf.nn.l2_normalize(vlad, axis=-1)
            return vlad
        else:
            # Placeholder for alternative modes if any
            return tf.reduce_mean(inputs, axis=[1, 2])

    def get_config(self):
        config = super(VladPooling, self).get_config()
        config.update({
            'k_centers': self.k_centers,
            'mode': self.mode
        })
        return config

    def compute_output_shape(self, input_shape):
        if self.mode == 'vlad':
            return (input_shape[0], self.k_centers * input_shape[-1])
        else:
            return (input_shape[0], input_shape[-1])

# ----------------------------
# Configuration Parameters
# ----------------------------
UNSEEN_DATA_DIR = "/Users/eliaskhoury/Desktop/Project490-SpeakerRecog/unseen_data"
MODEL_PATH = "/Users/eliaskhoury/Desktop/Project490-SpeakerRecog/final_finetuned_model.keras"
LABEL_ENCODER_PATH = "/Users/eliaskhoury/Desktop/Project490-SpeakerRecog/label_encoder.pkl"
SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0  # seconds
N_MFCC = 40
MAX_FRAMES = 250  # As per training configuration
BATCH_SIZE = 32
OUTPUT_PATH = "/Users/eliaskhoury/Desktop/Project490-SpeakerRecog/output_folder"
os.makedirs("output_folder", exist_ok=True)

# ----------------------------
# Load the Label Encoder
# ----------------------------
print(f"Attempting to load label encoder from: {LABEL_ENCODER_PATH}")

# List files in the label encoder directory
label_encoder_dir = os.path.dirname(LABEL_ENCODER_PATH)
print(f"Contents of the directory '{label_encoder_dir}':")
print(os.listdir(label_encoder_dir))

try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    class_labels = label_encoder.classes_
    print("Label encoder loaded successfully.")
    print("Classes:", class_labels)
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit(1)

# ----------------------------
# Load the Trained Model
# ----------------------------
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'VladPooling': VladPooling},
        compile=False
    )
    print("Model loaded successfully.")
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ----------------------------
# MFCC Extraction and Preprocessing
# ----------------------------
def extract_mfcc(y, sr, n_mfcc=40):
    """
    Extracts and normalizes MFCC features from an audio signal.
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = librosa.util.normalize(mfcc)
        return mfcc.T  # Shape: (time, n_mfcc)
    except Exception as e:
        print(f"  MFCC extraction error: {e}")
        return None

def preprocess_mfcc(mfcc, max_frames=250):
    """
    Pads or truncates MFCC features to ensure consistent input dimensions.
    """
    if mfcc is None:
        return None
    if mfcc.shape[0] > max_frames:
        mfcc = mfcc[:max_frames, :]
    else:
        padding = max_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')
    mfcc = np.expand_dims(mfcc, axis=-1)  # Shape: (max_frames, n_mfcc, 1)
    return mfcc

# ----------------------------
# Audio Chunking Function
# ----------------------------
def split_audio(y, sr, chunk_duration=5.0, overlap=0):
    """
    Splits audio into chunks with specified length and overlap.
    """
    total_samples = len(y)
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step = chunk_samples - overlap_samples
    chunks = []

    for start in range(0, total_samples, step):
        end = start + chunk_samples
        chunk = y[start:end]

        # If the chunk is shorter than the desired length, pad with zeros
        if len(chunk) < chunk_samples:
            padding = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, padding), 'constant')

        chunks.append(chunk)

    return chunks

# ----------------------------
# Add Counter-Speech Effects
# ----------------------------
def generate_adversarial_noise_frequency(audio_np, sr, output_path):
    """Generate targeted frequency perturbations to create adversarial noise."""
    # Apply Short-Time Fourier Transform (STFT) to convert to frequency domain
    stft = librosa.stft(audio_np, n_fft=512, hop_length=256, win_length=512)
    magnitude, phase = librosa.magphase(stft)

    # Add perturbations to the magnitude (distorting key frequencies in a critical band)
    perturbation = np.zeros(magnitude.shape)
    
    critical_bands = range(10, 100) # Example: Target specific bins
    perturbation[critical_bands, :] = np.random.normal(0, 3.4, (len(critical_bands), magnitude.shape[1]))
    
    # Add Temporal Variations
    temporal_variation = np.sin(np.linspace(0, 10 * np.pi, magnitude.shape[1])) + \
                     0.5 * np.cos(np.linspace(0, 5 * np.pi, magnitude.shape[1]))
    perturbation[critical_bands, :] += temporal_variation[np.newaxis, :]
    
        # Add subharmonics (half the frequency)
    for band in critical_bands:
        magnitude[band // 2, :] += 0.3 * magnitude[band, :]
        
    magnitude += perturbation

    # Reconstruct the audio from the perturbed magnitude and original phase
    perturbed_stft = magnitude * phase
    perturbed_audio = librosa.istft(perturbed_stft, hop_length=256, win_length=512)
    
    # Write the audio to a file
    sf.write(output_path, perturbed_audio, sr)
    
    return perturbed_audio

# ----------------------------
# Prediction Function
# ----------------------------
def predict_audio(audio_path, speaker_name):
    """
    Processes an audio file and predicts the speaker using majority voting across chunks.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)

        # Split into 5-second chunks
        chunks = split_audio(y, sr, chunk_duration=CHUNK_DURATION, overlap=0)

        if not chunks:
            print("    No valid chunks found.")
            return None
        
        # Create a subdirectory for the speaker in the output folder
        speaker_output_folder = os.path.join(OUTPUT_PATH, speaker_name)
        os.makedirs(speaker_output_folder, exist_ok=True)

        X = []
        for idx, chunk in enumerate(chunks, start=1):
            # Define a unique output path for each chunk within the speaker's folder
            output_path = os.path.join(speaker_output_folder, f"perturbed_audio_chunk_{idx}.wav")
            
            # Apply audio transformations
            transformed_chunk = generate_adversarial_noise_frequency(chunk, sr, output_path)
            
            # Extract MFCC features
            mfcc = extract_mfcc(transformed_chunk, sr, n_mfcc=N_MFCC)
            if mfcc is None:
                print(f"    Skipping chunk {idx} due to MFCC extraction error.")
                continue
            mfcc_processed = preprocess_mfcc(mfcc, max_frames=MAX_FRAMES)
            if mfcc_processed is None:
                print(f"    Skipping chunk {idx} due to preprocessing error.")
                continue
            X.append(mfcc_processed)

        if not X:
            print("    No valid MFCCs extracted.")
            return None

        X = np.array(X)  # Shape: (num_chunks, max_frames, n_mfcc, 1)

        # Predict on all chunks
        preds = model.predict(X, batch_size=BATCH_SIZE)

        # Get the predicted class for each chunk
        predicted_indices = np.argmax(preds, axis=1)

        # Majority voting
        counts = np.bincount(predicted_indices, minlength=len(class_labels))
        top_idx = np.argmax(counts)
        top_class = class_labels[top_idx]
        confidence = counts[top_idx] / len(predicted_indices)

        return top_class, confidence

    except Exception as e:
        print(f"  Error processing {audio_path}: {e}")
        return None

# ----------------------------
# Main Function
# ----------------------------
def main():
    """
    Iterates through the unseen data directory, processes each audio file, and prints predictions.
    """
    if not os.path.exists(UNSEEN_DATA_DIR):
        print(f"Unseen data directory not found: {UNSEEN_DATA_DIR}")
        exit(1)

    speakers = sorted([d for d in os.listdir(UNSEEN_DATA_DIR) if os.path.isdir(os.path.join(UNSEEN_DATA_DIR, d))])

    for speaker in speakers:
        speaker_dir = os.path.join(UNSEEN_DATA_DIR, speaker)
        print(f"\nProcessing speaker folder: {speaker}")

        audio_files = sorted([f for f in os.listdir(speaker_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))])

        if not audio_files:
            print(f"  No valid audio files found in {speaker_dir}.")
            continue

        for audio_file in tqdm(audio_files, desc=f"  Analyzing audios for speaker '{speaker}'", unit="file", disable=True):
            audio_path = os.path.join(speaker_dir, audio_file)
            print(f"    Analyzing {audio_file}...")
            prediction = predict_audio(audio_path, speaker)

            if prediction is None:
                print("      Prediction skipped due to errors.")
                continue

            top_class, confidence = prediction
            print(f"      Predicted Speaker: {top_class} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
