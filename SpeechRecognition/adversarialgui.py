import os
import whisper
import pyaudio
import numpy as np
import queue
import threading
import torch
import warnings
from scipy.io.wavfile import write
import librosa
from jiwer import wer  # Import jiwer for WER calculation
import soundfile as sf  # Import soundfile for saving audio
import tkinter as tk
from tkinter import scrolledtext, messagebox

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
        self.root.title("Adversarial Audio Transcription and WER Calculator")
        self.root.geometry("900x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize GUI Components
        self.create_widgets()

        # Initialize variables
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_data = []  # List to store raw audio data
        self.transcribed_text = ""
        self.reference_text = ""

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None

        # Load Whisper model in a separate thread to prevent GUI freezing
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Reference Text Label and Entry
        ref_label = tk.Label(self.root, text="Enter Reference Text:", font=("Helvetica", 14))
        ref_label.pack(pady=10)

        self.ref_entry = scrolledtext.ScrolledText(self.root, height=4, width=80, font=("Helvetica", 12))
        self.ref_entry.pack(pady=5)

        # Record Button
        self.record_button = tk.Button(self.root, text="Start Recording", command=self.start_recording, bg="#4CAF50",
                                       fg="white", font=("Helvetica", 14))
        self.record_button.pack(pady=10)

        # Stop Button
        self.stop_button = tk.Button(self.root, text="Stop Recording", command=self.stop_recording, bg="#f44336",
                                     fg="white", font=("Helvetica", 14), state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        # Transcribed Text Label and Display
        trans_label = tk.Label(self.root, text="Transcribed Text:", font=("Helvetica", 14))
        trans_label.pack(pady=10)

        self.trans_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Helvetica", 12), height=10, width=80)
        self.trans_text.pack(padx=20, pady=5)
        self.trans_text.config(state=tk.DISABLED)  # Make read-only

        # WER Label and Display
        wer_label = tk.Label(self.root, text="Word Error Rate (WER):", font=("Helvetica", 14))
        wer_label.pack(pady=10)

        self.wer_display = tk.Label(self.root, text="N/A", font=("Helvetica", 14), fg="#2E86C1")
        self.wer_display.pack(pady=5)

        # Status Label
        self.status_label = tk.Label(self.root, text="Loading Whisper model...", font=("Helvetica", 12), fg="#E67E22")
        self.status_label.pack(pady=10)

    def load_model(self):
        try:
            self.update_status("Loading Whisper model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
            self.model = whisper.load_model(MODEL_SIZE).to(self.device)
            self.update_status(f"Whisper model '{MODEL_SIZE}' loaded on {self.device}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Whisper model: {e}")
            self.update_status(f"Error loading model: {e}")

    def start_recording(self):
        if not hasattr(self, 'model'):
            messagebox.showwarning("Model Loading", "Whisper model is still loading. Please wait.")
            return

        # Get reference text
        self.reference_text = self.ref_entry.get("1.0", tk.END).strip()
        if not self.reference_text:
            messagebox.showwarning("Input Required", "Please enter the reference text before recording.")
            return

        self.is_recording = True
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.trans_text.config(state=tk.NORMAL)
        self.trans_text.delete(1.0, tk.END)  # Clear previous transcription
        self.wer_display.config(text="N/A")
        self.audio_data = []  # Reset accumulated audio

        # Open the audio stream
        try:
            self.stream = self.p.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=SAMPLE_RATE,
                                      input=True,
                                      frames_per_buffer=int(SAMPLE_RATE * (CHUNK_DURATION - OVERLAP)),
                                      stream_callback=self.audio_callback)
        except Exception as e:
            messagebox.showerror("Audio Stream Error", f"Could not open audio stream: {e}")
            self.stop_recording()
            return

        self.stream.start_stream()

        self.update_status("Recording...")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Stop and close the audio stream
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Send exit signal to the audio callback
        self.audio_queue.put(None)

        self.update_status("Recording stopped. Processing...")

        # Start processing in a separate thread to avoid blocking the GUI
        processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        processing_thread.start()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        This function is called by PyAudio whenever there's new audio data.
        It puts the raw audio data into a queue for processing.
        """
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def generate_adversarial_noise_frequency(self, audio_np):
        """Generate targeted frequency perturbations to create adversarial noise."""
        # Apply Short-Time Fourier Transform (STFT) to convert to frequency domain
        stft = librosa.stft(audio_np, n_fft=512, hop_length=256, win_length=512)
        magnitude, phase = librosa.magphase(stft)

        # Add perturbations to the magnitude (distorting key frequencies in a critical band)
        perturbation = np.zeros(magnitude.shape)
        
        critical_bands = range(10, 100)  # Example: Target specific bins
        perturbation[critical_bands, :] = np.random.normal(0, 3.4, (len(critical_bands), magnitude.shape[1]))
        
        # Add Temporal Variations
        temporal_variation = np.sin(np.linspace(0, 10 * np.pi, magnitude.shape[1])) + \
                             0.5 * np.cos(np.linspace(0, 5 * np.pi, magnitude.shape[1]))
        perturbation[critical_bands, :] += temporal_variation[np.newaxis, :]
        
        # Add subharmonics (half the frequency)
        for band in critical_bands:
            if band // 2 < magnitude.shape[0]:
                magnitude[band // 2, :] += 0.3 * magnitude[band, :]
        
        magnitude += perturbation

        # Reconstruct the audio from the perturbed magnitude and original phase
        perturbed_stft = magnitude * phase
        perturbed_audio = librosa.istft(perturbed_stft, hop_length=256, win_length=512)

        return perturbed_audio

    def process_audio(self):
        """Process the recorded audio: apply adversarial noise, save, transcribe, and calculate WER."""
        try:
            # Collect all audio data from the queue
            audio_data = []
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                if chunk is None:
                    break
                audio_data.append(chunk)

            if not audio_data:
                self.update_status("No audio data recorded.")
                return

            # Combine all chunks into a single bytes object
            raw_audio = b''.join(audio_data)

            # Convert raw audio data to numpy array
            audio_np = np.frombuffer(raw_audio, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Apply adversarial frequency perturbations
            perturbed_audio = self.generate_adversarial_noise_frequency(audio_np)

            # Save the complete adversarial audio to a .wav file
            try:
                # Normalize the audio to prevent clipping
                normalized_audio = perturbed_audio / np.max(np.abs(perturbed_audio))
                sf.write(OUTPUT_FILE, normalized_audio, SAMPLE_RATE)
                self.update_status(f"Adversarial audio saved as {OUTPUT_FILE}. Transcribing...")
            except Exception as e:
                self.update_status(f"Error saving audio: {e}")
                messagebox.showerror("Save Error", f"An error occurred while saving audio: {e}")
                return

            # Perform transcription using Whisper
            try:
                result = self.model.transcribe(perturbed_audio, fp16=(self.device == "cuda"), language='en')
                self.transcribed_text = result['text'].replace('\n', ' ').strip()

                # Update transcription display
                self.trans_text.config(state=tk.NORMAL)
                self.trans_text.insert(tk.END, self.transcribed_text)
                self.trans_text.config(state=tk.DISABLED)

                # Calculate WER
                calculated_wer = wer(self.reference_text, self.transcribed_text)
                self.wer_display.config(text=f"{calculated_wer:.2%}")

                self.update_status("Transcription and WER calculation completed.")
            except Exception as e:
                self.update_status(f"Error during transcription: {e}")
                messagebox.showerror("Transcription Error", f"An error occurred during transcription: {e}")

        except Exception as e:
            self.update_status(f"Error during processing: {e}")
            messagebox.showerror("Processing Error", f"An error occurred during processing: {e}")

    def update_status(self, message):
        """Update the status label in the GUI."""
        self.status_label.config(text=message)

    def on_close(self):
        """Handle the window closing event."""
        if self.is_recording:
            if messagebox.askokcancel("Quit", "Recording is in progress. Do you want to stop and exit?"):
                self.is_recording = False
                if self.stream is not None:
                    self.stream.stop_stream()
                    self.stream.close()
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
