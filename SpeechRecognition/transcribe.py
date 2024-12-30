import os
import whisper

# Load the Whisper model
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Paths for input and output
input_dir = "input"
output_dir = "output"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each audio file in the input directory
for audio_file in os.listdir(input_dir):
    if audio_file.endswith((".mp3", ".wav", ".mp4")):  # Check for supported formats
        print(f"Transcribing {audio_file}...")
        file_path = os.path.join(input_dir, audio_file)

        # Perform transcription
        result = model.transcribe(file_path)

        # Save the transcription to a text file
        output_path = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Saved transcription to {output_path}")
