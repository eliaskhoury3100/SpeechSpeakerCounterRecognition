import os
import whisper
import pyaudio
import numpy as np
import queue
import threading
import time
import torch
import warnings
import tkinter as tk
from tkinter import scrolledtext, messagebox
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

class AdversarialTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adversarial Real-Time Transcription")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize GUI Components
        self.create_widgets()

        # Initialize variables
        self.is_transcribing = False
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.stream = None
        self.audio_data_accumulated = []

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # Initialize the exit flag
        self.exit_flag = False

        # Load Whisper model in a separate thread to prevent GUI freezing
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Start Button
        self.start_button = tk.Button(
            self.root, 
            text="Start Transcription", 
            command=self.start_transcription, 
            bg="#4CAF50", 
            fg="white", 
            font=("Helvetica", 14)
        )
        self.start_button.pack(pady=10)

        # Stop Button
        self.stop_button = tk.Button(
            self.root, 
            text="Stop Transcription", 
            command=self.stop_transcription, 
            bg="#f44336", 
            fg="white", 
            font=("Helvetica", 14), 
            state=tk.DISABLED
        )
        self.stop_button.pack(pady=10)

        # Scrolled Text for Transcription Display
        self.text_area = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            font=("Helvetica", 12)
        )
        self.text_area.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        self.text_area.config(state=tk.DISABLED)  # Make read-only

    def load_model(self):
        try:
            self.update_text("Loading Whisper model...\n")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
            self.model = whisper.load_model(MODEL_SIZE).to(self.device)
            self.update_text(f"Whisper model '{MODEL_SIZE}' loaded on {self.device}.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Whisper model: {e}")
            self.update_text(f"Error loading model: {e}\n")

    def start_transcription(self):
        if not hasattr(self, 'model'):
            messagebox.showwarning("Model Loading", "Whisper model is still loading. Please wait.")
            return

        self.is_transcribing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)  # Clear previous text
        self.text_area.config(state=tk.DISABLED)

        # Initialize Whisper transcription variables
        self.previous_chunk = np.zeros(int(SAMPLE_RATE * OVERLAP), dtype=np.float32)
        self.fp16_flag = self.device == "cuda"

        # Calculate frames per buffer
        frames_per_buffer = int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP))

        # Open the audio stream
        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self.audio_callback
            )
        except Exception as e:
            messagebox.showerror("Audio Stream Error", f"Could not open audio stream: {e}")
            self.stop_transcription()
            return

        self.stream.start_stream()

        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        self.transcription_thread.start()

        # Reset accumulated audio data
        self.audio_data_accumulated = []

        self.update_text("Transcription started...\n")

    def stop_transcription(self):
        if not self.is_transcribing:
            return

        self.is_transcribing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Stop and close the audio stream
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Send exit signal to the transcription thread
        self.audio_queue.put(None)
        self.transcription_thread.join()

        # Save the complete adversarial audio to a .wav file after transcription
        if self.audio_data_accumulated:
            full_audio = np.concatenate(self.audio_data_accumulated)
            try:
                write(OUTPUT_FILE, SAMPLE_RATE, np.clip(full_audio * 32768, -32768, 32767).astype(np.int16))
                self.update_text(f"\nTranscription stopped. The complete adversarial audio is saved in: {OUTPUT_FILE}\n")
            except Exception as e:
                self.update_text(f"\nError saving audio: {e}\n")
        else:
            self.update_text("\nNo audio data to save.\n")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        This function is called by PyAudio whenever there's new audio data.
        It puts the raw audio data into a queue for processing.
        """
        if self.is_transcribing:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def generate_adversarial_noise_frequency(self, audio_np):
        """Generate targeted frequency perturbations to create adversarial noise."""
        try:
            # Apply Short-Time Fourier Transform (STFT) to convert to frequency domain
            stft = librosa.stft(audio_np, n_fft=512, hop_length=256, win_length=512)
            magnitude, phase = librosa.magphase(stft)

            # Add perturbations to the magnitude (distorting key frequencies)
            perturbation = np.random.normal(0, 0.25, magnitude.shape)  # Perturb with random noise
            magnitude += perturbation

            # Reconstruct the audio from the perturbed magnitude and original phase
            perturbed_stft = magnitude * phase
            perturbed_audio = librosa.istft(perturbed_stft, hop_length=256, win_length=512)

            # Ensure the perturbed audio has the same length as the input
            if len(perturbed_audio) < len(audio_np):
                perturbed_audio = np.pad(perturbed_audio, (0, len(audio_np) - len(perturbed_audio)), mode='constant')
            else:
                perturbed_audio = perturbed_audio[:len(audio_np)]

            return perturbed_audio
        except Exception as e:
            self.update_text(f"\nError generating adversarial noise: {e}\n")
            return audio_np  # Return original audio if perturbation fails

    def transcribe_audio(self):
        """
        This function runs in a separate thread.
        It continuously reads audio data from the queue, processes it, and transcribes it.
        """
        try:
            while self.is_transcribing:
                # Wait for the next chunk of audio data
                audio_data = self.audio_queue.get()

                if audio_data is None:
                    break  # Exit signal

                # Convert raw audio data to numpy array
                audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

                # Apply adversarial frequency perturbations
                perturbed_audio = self.generate_adversarial_noise_frequency(audio_np)

                # Accumulate the perturbed audio
                self.audio_data_accumulated.append(perturbed_audio)

                # Combine with previous chunk for overlap
                combined_audio = np.concatenate((self.previous_chunk, perturbed_audio))

                # Perform transcription using Whisper
                result = self.model.transcribe(combined_audio, fp16=self.fp16_flag, language='en')
                transcribed_text = result['text'].replace('\n', ' ').strip()

                # Update the text area with the transcribed text
                if transcribed_text:
                    self.update_text(transcribed_text + " ")

                # Save the last portion of this chunk for overlap
                self.previous_chunk = perturbed_audio[-int(SAMPLE_RATE * OVERLAP):]

        except Exception as e:
            self.update_text(f"\nError during transcription: {e}\n")

    def update_text(self, text):
        """Thread-safe method to update the text area."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)  # Auto-scroll to the end
        self.text_area.config(state=tk.DISABLED)

    def on_close(self):
        """Handle the window closing event."""
        if self.is_transcribing:
            if messagebox.askokcancel("Quit", "Transcription is running. Do you want to stop and exit?"):
                self.stop_transcription()
                self.p.terminate()
                self.root.destroy()
        else:
            self.p.terminate()
            self.root.destroy()

# ----------------------------
# Main Function
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AdversarialTranscriptionApp(root)
    root.mainloop()