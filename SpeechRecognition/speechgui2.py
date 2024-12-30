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
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1
FORMAT = pyaudio.paInt16

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription and WER Calculator")
        self.root.geometry("800x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize GUI Components
        self.create_widgets()

        # Initialize variables
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcription_thread = None
        self.recording_thread = None
        self.audio_data = b''
        self.transcribed_text = ""
        self.reference_text = ""

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # Load Whisper model in a separate thread to prevent GUI freezing
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        # Reference Text Label and Entry
        ref_label = tk.Label(self.root, text="Enter Reference Text:", font=("Helvetica", 14))
        ref_label.pack(pady=10)

        self.ref_entry = tk.Text(self.root, height=4, width=80, font=("Helvetica", 12))
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

        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()

        self.update_status("Recording...")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        self.update_status("Recording stopped. Transcribing...")

    def record_audio(self):
        try:
            # Open the audio stream
            stream = self.p.open(format=FORMAT,
                                 channels=CHANNELS,
                                 rate=SAMPLE_RATE,
                                 input=True,
                                 frames_per_buffer=1024)

            frames = []
            while self.is_recording:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)

            # Stop and close the stream
            stream.stop_stream()
            stream.close()

            self.audio_data = b''.join(frames)

            # Start transcription in the main thread
            self.root.after(0, self.transcribe_audio)

        except Exception as e:
            self.update_status(f"Error during recording: {e}")
            messagebox.showerror("Recording Error", f"An error occurred while recording: {e}")
            self.is_recording = False
            self.record_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def transcribe_audio(self):
        try:
            if not self.audio_data:
                self.update_status("No audio data to transcribe.")
                return

            # Convert raw audio data to numpy array
            audio_np = np.frombuffer(self.audio_data, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Whisper expects audio as a 1D float32 numpy array
            result = self.model.transcribe(audio_np, fp16=(self.device == "cuda"), language='en')
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

    def update_status(self, message):
        self.status_label.config(text=message)

    def on_close(self):
        """Handle the window closing event."""
        if self.is_recording:
            if messagebox.askokcancel("Quit", "Recording is in progress. Do you want to stop and exit?"):
                self.is_recording = False
                self.recording_thread.join()
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
    app = TranscriptionApp(root)
    root.mainloop()
