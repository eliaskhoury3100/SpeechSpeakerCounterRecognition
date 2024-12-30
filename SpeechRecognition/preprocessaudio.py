import os
import librosa
import soundfile as sf
import numpy as np

# Path to the folder containing the speaker folders
data_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\data'  
processed_data_dir = r'C:\Users\AUB\OneDrive - American University of Beirut\Desktop\counter_speech_project\speaker-recognition\processed_data'

# Create the 'processed_data' folder if it doesn't exist
os.makedirs(processed_data_dir, exist_ok=True)

def split_audio(audio, chunk_length=5, overlap=0, sr=16000):
    """
    Splits audio into chunks with specified length and overlap.

    Parameters:
    - audio (np.ndarray): The audio signal.
    - chunk_length (float): Length of each chunk in seconds.
    - overlap (float): Overlap between chunks in seconds.
    - sr (int): Sampling rate.

    Returns:
    - List[np.ndarray]: List of audio chunks.
    """
    total_samples = len(audio)
    chunk_samples = int(chunk_length * sr)
    overlap_samples = int(overlap * sr)
    step = chunk_samples - overlap_samples
    chunks = []

    for start in range(0, total_samples, step):
        end = start + chunk_samples
        chunk = audio[start:end]

        # If the chunk is shorter than the desired length, pad with zeros
        if len(chunk) < chunk_samples:
            padding = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, padding), 'constant')

        chunks.append(chunk)

    return chunks

# Loop through each speaker folder
for speaker in os.listdir(data_dir):
    speaker_dir = os.path.join(data_dir, speaker)
    
    # Skip if it's not a directory
    if not os.path.isdir(speaker_dir):
        continue
    
    # Create a corresponding folder in the processed_data directory
    processed_speaker_dir = os.path.join(processed_data_dir, speaker)
    os.makedirs(processed_speaker_dir, exist_ok=True)
    
    # Initialize a counter for processed audio files
    processed_count = 0
    max_audios = 30  # Maximum number of audios to process per speaker
    
    # Loop through each audio file in the speaker folder
    for audio_file in os.listdir(speaker_dir):
        # Check if the maximum number of audios has been processed
        if processed_count >= max_audios:
            print(f"Reached {max_audios} audios for speaker '{speaker}'. Skipping remaining files.")
            break  # Exit the loop for this speaker
        
        audio_path = os.path.join(speaker_dir, audio_file)
        
        # Skip if it's not a valid audio file
        if not audio_file.lower().endswith(('.wav', '.flac', '.mp3', '.m4a')):
            continue
        
        try:
            # Load the audio file
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Normalize the audio
            audio = librosa.util.normalize(audio)
            
            # Trim leading and trailing silence
            audio, _ = librosa.effects.trim(audio)
            
            # Split audio into 5-second chunks with 0-second overlap
            audio_chunks = split_audio(audio, chunk_length=5, overlap=0, sr=16000)
            
            # Save each chunk as a separate file
            for i, chunk in enumerate(audio_chunks):
                chunk_output_path = os.path.join(
                    processed_speaker_dir, 
                    f"{os.path.splitext(audio_file)[0]}_chunk{i}.wav"
                )
                sf.write(chunk_output_path, chunk, sr)
                print(f"Processed and saved: {chunk_output_path}")
            
            # Increment the processed audio counter
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {audio_file} for speaker '{speaker}': {e}")
            continue  # Skip to the next file if there's an error

    print(f"Finished processing speaker '{speaker}'. Total audios processed: {processed_count}")

print("Audio preprocessing complete!")
