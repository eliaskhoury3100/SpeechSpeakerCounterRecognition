import os
import whisper
import pyaudio
import numpy as np
import queue
import threading
import time
import torch
import warnings
from scipy.io.wavfile import write
import librosa

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large
CHUNK_DURATION = 2  # seconds
OVERLAP = 0.2  # seconds of overlap
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1
FORMAT = pyaudio.paInt16
OUTPUT_FILE = "adversarial_output.wav"  # Complete output file for adversarial audio

# Initialize Whisper model
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
model = whisper.load_model(MODEL_SIZE).to(device)
print(f"Whisper model '{MODEL_SIZE}' loaded on {device}.")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Create a queue to communicate between the audio thread and main thread
audio_queue = queue.Queue()

# Calculate the number of frames per buffer
frames_per_buffer = int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP))

def generate_adversarial_noise_frequency(audio_np):
    """Generate targeted frequency perturbations to create adversarial noise."""
    # Apply Short-Time Fourier Transform (STFT) to convert to frequency domain
    stft = librosa.stft(audio_np, n_fft=512, hop_length=256, win_length=512)
    magnitude, phase = librosa.magphase(stft)

    # Add perturbations to the magnitude (distorting key frequencies)
    perturbation = np.random.normal(0, 0.9, magnitude.shape)  # Perturb with random noise
    magnitude += perturbation

    # Reconstruct the audio from the perturbed magnitude and original phase
    perturbed_stft = magnitude * phase
    perturbed_audio = librosa.istft(perturbed_stft, hop_length=256, win_length=512)
    
    return perturbed_audio

def audio_callback(in_data, frame_count, time_info, status):
    """
    This function is called by PyAudio whenever there's new audio data.
    It puts the raw audio data into a queue for processing.
    """
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# Open the audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=frames_per_buffer,
                stream_callback=audio_callback)

print("Starting audio stream...")
stream.start_stream()

def transcribe_audio():
    """
    This function runs in a separate thread.
    It continuously reads audio data from the queue, processes it, and transcribes it.
    """
    previous_chunk = np.zeros(int(SAMPLE_RATE * OVERLAP), dtype=np.float32)
    fp16_flag = device == "cuda"
    
    audio_data_accumulated = []  # List to accumulate audio chunks
    
    while True:
        try:
            # Wait for the next chunk of audio data
            audio_data = audio_queue.get()

            if audio_data is None:
                break  # Exit signal

            # Convert raw audio data to numpy array
            audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Apply adversarial frequency perturbations
            perturbed_audio = generate_adversarial_noise_frequency(audio_np)

            # Accumulate the perturbed audio
            audio_data_accumulated.append(perturbed_audio)

            # Perform transcription using Whisper
            result = model.transcribe(perturbed_audio, fp16=fp16_flag, language='en')
            transcribed_text = result['text'].replace('\n', ' ').strip()

            # Print the transcription without adding a new line
            print(transcribed_text, end=' ', flush=True)

            # Save the last portion of this chunk for overlap
            previous_chunk = perturbed_audio[-int(SAMPLE_RATE * OVERLAP):]

        except Exception as e:
            print(f"Error during transcription: {e}")

    # Save the complete adversarial audio to a .wav file after transcription
    full_audio = np.concatenate(audio_data_accumulated)
    write(OUTPUT_FILE, SAMPLE_RATE, (full_audio * 32768).astype(np.int16))  # Scale back to int16

# Start the transcription thread
transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
transcription_thread.start()

try:
    print("Press Ctrl+C to stop the transcription.")
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopping transcription...")

# Stop and close the audio stream
stream.stop_stream()
stream.close()
p.terminate()

# Send exit signal to the transcription thread
audio_queue.put(None)
transcription_thread.join()

print("Transcription stopped. The complete adversarial audio is saved in:", OUTPUT_FILE)
