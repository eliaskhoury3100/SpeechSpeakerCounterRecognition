import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
import joblib
import sounddevice as sd
import soundfile as sf
import time
import noisereduce as nr  # Install via pip install noisereduce
import webrtcvad            # Install via pip install webrtcvad
import matplotlib.pyplot as plt
import librosa.display

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
MODEL_PATH = r"C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\final_finetuned_model.keras"
LABEL_ENCODER_PATH = r"C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\label_encoder.pkl"
SAMPLE_RATE = 16000
RECORD_DURATION = 7.0  # seconds to record
TRIM_DURATION = 5.0    # seconds to trim to
N_MFCC = 40
MAX_FRAMES = 250
BATCH_SIZE = 32
N_FFT = 2048           # Default value used in training
HOP_LENGTH = 512       # Default value used in training

# ----------------------------
# Load the Label Encoder
# ----------------------------
print(f"Loading label encoder from: {LABEL_ENCODER_PATH}")
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
def extract_mfcc(y, sr, n_mfcc=40, n_fft=2048, hop_length=512, save_path=None):
    """
    Extracts and normalizes MFCC features from an audio signal.
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = librosa.util.normalize(mfcc)  # Normalize MFCC across all axes (axis=None)
        mfcc = mfcc.T  # Shape: (time, n_mfcc)
        if save_path:
            np.save(save_path, mfcc)
            print(f"MFCC saved to {save_path}")
        return mfcc
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

def reduce_noise(y, sr, noise_profile):
    """
    Applies noise reduction to the audio signal using a pre-recorded noise profile.
    """
    try:
        y_denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_profile)
        return y_denoised
    except Exception as e:
        print(f"  Noise reduction error: {e}")
        return y  # Return original if noise reduction fails

def apply_vad(y, sr, frame_duration=30, padding_duration=300, mode=3):
    """
    Applies Voice Activity Detection to retain only speech segments.
    Converts audio from float32 to int16 as required by webrtcvad.
    """
    try:
        vad = webrtcvad.Vad(mode)
        frame_length = int(sr * frame_duration / 1000)  # Frame length in samples (e.g., 480 for 30ms at 16kHz)
        padding_length = int(sr * padding_duration / 1000)  # Not used in current implementation

        # Split audio into frames
        frames = []
        for start in range(0, len(y), frame_length):
            end = start + frame_length
            frame = y[start:end]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
            
            # Convert float32 to int16
            frame_int16 = (frame * 32767).astype(np.int16)
            is_speech = vad.is_speech(frame_int16.tobytes(), sample_rate=sr)
            if is_speech:
                frames.append(frame)

        if not frames:
            print("    No speech detected after VAD.")
            return y  # If no speech detected, return original audio

        # Concatenate speech frames
        y_vad = np.concatenate(frames)
        return y_vad

    except Exception as e:
        print(f"  VAD processing error: {e}")
        raise e  # Re-raise the exception to be handled in predict_audio

def dynamic_range_compression(y, coef=0.97):
    """
    Applies pre-emphasis to the audio signal.
    """
    return librosa.effects.preemphasis(y, coef=coef)

# ----------------------------
# Record Noise Profile (Once)
# ----------------------------
def record_noise_profile(duration=2.0, sr=16000):
    """
    Records a short audio clip to capture the ambient noise profile.
    """
    print(f"Recording {duration} seconds of ambient noise for noise reduction...")
    noise = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    noise = np.squeeze(noise)
    print("Noise recording complete.")
    return noise

# ----------------------------
# Save and Compare MFCCs for Debugging
# ----------------------------
def plot_mfcc_comparison(real_time_mfcc_path, offline_mfcc_path):
    """
    Plots and compares MFCCs from real-time and offline audio.
    """
    real_time_mfcc = np.load(real_time_mfcc_path)
    offline_mfcc = np.load(offline_mfcc_path)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(real_time_mfcc.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time')
    plt.title('Real-Time Audio MFCC')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(offline_mfcc.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time')
    plt.title('Offline Audio MFCC')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_audio(y, sr, noise_profile, offline_mfcc_path=None):
    """
    Processes a single 5-second audio chunk and predicts the speaker.
    Optionally saves MFCC for comparison with offline.
    """
    try:
        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y)
        print("Silence trimmed.")

        # Apply VAD
        y = apply_vad(y, sr)
        print("Voice Activity Detection applied.")

        # Apply noise reduction
        y = reduce_noise(y, sr, noise_profile)
        print("Noise reduction applied.")

        # Apply dynamic range compression
        y = dynamic_range_compression(y)
        print("Dynamic range compression applied.")

        # Debug: Check the length after processing
        trimmed_duration = len(y) / sr
        print(f"Trimmed audio duration after processing: {trimmed_duration:.2f} seconds")

        # If trimmed audio is longer than TRIM_DURATION, extract the first TRIM_DURATION seconds
        if trimmed_duration > TRIM_DURATION:
            end_sample = int(TRIM_DURATION * sr)
            y = y[:end_sample]
            print(f"Audio trimmed to first {TRIM_DURATION} seconds.")
        elif trimmed_duration < TRIM_DURATION:
            # Pad with zeros if shorter
            padding = int(TRIM_DURATION * sr) - len(y)
            y = np.pad(y, (0, padding), 'constant')
            print(f"Audio padded to reach {TRIM_DURATION} seconds.")

        # Extract MFCC
        if offline_mfcc_path:
            mfcc = extract_mfcc(y, sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, save_path='real_time_mfcc.npy')
        else:
            mfcc = extract_mfcc(y, sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        if mfcc is None:
            print("    Skipping chunk due to MFCC extraction error.")
            return None

        # Preprocess MFCC
        mfcc_processed = preprocess_mfcc(mfcc, max_frames=MAX_FRAMES)
        if mfcc_processed is None:
            print("    Skipping chunk due to preprocessing error.")
            return None

        # Expand dimensions to match model input
        X = np.expand_dims(mfcc_processed, axis=0)  # Shape: (1, max_frames, n_mfcc, 1)
        print(f"MFCC shape for prediction: {X.shape}")

        # Predict
        preds = model.predict(X, batch_size=BATCH_SIZE)
        predicted_idx = np.argmax(preds, axis=1)[0]
        top_class = class_labels[predicted_idx]
        confidence = preds[0, predicted_idx]
        print(f"Model prediction confidence scores: {preds}")
        print(f"Predicted Speaker Index: {predicted_idx}, Speaker: {top_class}, Confidence: {confidence:.2f}")

        # Optional: Plot MFCC comparison
        if offline_mfcc_path:
            plot_mfcc_comparison('real_time_mfcc.npy', offline_mfcc_path)

        return top_class, confidence

    except Exception as e:
        print(f"  Error during prediction: {e}")
        return None

# ----------------------------
# Real-time Audio Capture and Prediction
# ----------------------------
def real_time_prediction(noise_profile, offline_mfcc_path=None):
    """
    Captures audio in real-time, processes it, and makes a single prediction.
    Optionally compares MFCCs with offline.
    """
    try:
        print(f"Recording {RECORD_DURATION} seconds of audio...")
        # Record audio
        audio = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")

        # Flatten the audio array
        audio = np.squeeze(audio)
        print(f"Recorded audio shape: {audio.shape}")

        # Predict
        prediction = predict_audio(audio, SAMPLE_RATE, noise_profile, offline_mfcc_path=offline_mfcc_path)
        if prediction is not None:
            top_class, confidence = prediction
            print(f"\nPredicted Speaker: {top_class} (Confidence: {confidence:.2f})")
        else:
            print("Prediction could not be made due to preprocessing errors.")

    except Exception as e:
        print(f"An error occurred during real-time prediction: {e}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    """
    Entry point for the script.
    """
    # Record noise profile once (you can save and reuse this profile)
    noise_profile = record_noise_profile(duration=2.0, sr=SAMPLE_RATE)

    # Path to an offline MFCC for comparison (optional)
    # Ensure you have an offline MFCC saved, e.g., 'offline_mfcc.npy'
    offline_mfcc_path = None  # Set to 'path_to_offline_mfcc.npy' if available

    # Perform real-time prediction
    real_time_prediction(noise_profile, offline_mfcc_path=offline_mfcc_path)

if __name__ == "__main__":
    main()
