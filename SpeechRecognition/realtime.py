import os
import whisper
import pyaudio
import numpy as np
import queue
import threading
import time
import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large
CHUNK_DURATION = 2  # seconds
OVERLAP = 0.1  # seconds of overlap
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1
FORMAT = pyaudio.paInt16

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
    
    while True:
        try:
            # Wait for the next chunk of audio data
            audio_data = audio_queue.get()

            if audio_data is None:
                break  # Exit signal

            # Convert raw audio data to numpy array
            audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Combine with previous chunk for overlap
            audio_np = np.concatenate((previous_chunk, audio_np))

            # Perform transcription using Whisper
            # Ensure that the input shape is correct (e.g., [1, n_samples] if required)
            result = model.transcribe(audio_np, fp16=fp16_flag, language='en')
            transcribed_text = result['text'].replace('\n', ' ').strip()

            # Print the transcription without adding a new line
            print(transcribed_text, end=' ', flush=True)

            # Save the last portion of this chunk for overlap
            previous_chunk = audio_np[-int(SAMPLE_RATE * OVERLAP):]

        except Exception as e:
            print(f"\nError during transcription: {e}")
            break  # Optionally exit the loop on error

# Start the transcription thread without daemon=True
transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.start()

try:
    print("\nPress Ctrl+C to stop the transcription.")
    while transcription_thread.is_alive():
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

print("Transcription stopped.")
