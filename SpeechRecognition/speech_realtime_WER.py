import os
import whisper
import pyaudio
import numpy as np
import queue
import threading
import time
import torch
import warnings
from jiwer import wer  # Import jiwer for WER calculation

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 10  # Maximum recording duration (adjust as needed)

# Initialize Whisper model
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
model = whisper.load_model(MODEL_SIZE).to(device)
print(f"Whisper model '{MODEL_SIZE}' loaded on {device}.")

# Initialize PyAudio
p = pyaudio.PyAudio()

def record_audio():
    """
    Records audio from the microphone until the user stops it.
    Returns the recorded audio data as bytes.
    """
    print("\nPlease prepare to speak. Press Enter to start recording.")
    input()  # Wait for user to press Enter to start
    print("Recording... Press Enter to stop recording.")

    # Open the audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    recording = True

    def stop_recording():
        nonlocal recording
        input()  # Wait for user to press Enter to stop
        recording = False

    # Start a thread to listen for the stop signal
    stop_thread = threading.Thread(target=stop_recording)
    stop_thread.start()

    while recording:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"\nError while recording: {e}")
            break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording stopped.")
    return b''.join(frames)

def transcribe_audio(audio_data):
    """
    Transcribes the given audio data using Whisper.
    Returns the transcribed text.
    """
    # Convert raw audio data to numpy array
    audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

    # Whisper expects audio as a 1D float32 numpy array
    result = model.transcribe(audio_np, fp16=(device == "cuda"), language='en')
    transcribed_text = result['text'].replace('\n', ' ').strip()
    return transcribed_text

def main():
    # Step 1: Get reference text from the user
    reference_text = input("Enter the text you are going to say: ").strip()
    if not reference_text:
        print("Reference text cannot be empty. Exiting.")
        return

    # Step 2: Record audio from the user
    audio_data = record_audio()
    if not audio_data:
        print("No audio data recorded. Exiting.")
        return

    # Step 3: Transcribe the recorded audio
    print("\nTranscribing audio...")
    transcribed_text = transcribe_audio(audio_data)
    print(f"\nTranscribed Text:\n{transcribed_text}")

    # Step 4: Calculate Word Error Rate (WER)
    calculated_wer = wer(reference_text, transcribed_text)
    print(f"\nWord Error Rate (WER): {calculated_wer:.2%}")

if __name__ == "__main__":
    main()
