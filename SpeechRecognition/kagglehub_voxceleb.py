import random

# Input metadata file and output for filtered list
input_file = "iden_split.txt"  # Replace with the actual path
output_file = "filtered_files.txt"

# Number of speakers to select
num_speakers = 10

# Read the metadata
with open(input_file, "r") as f:
    lines = f.readlines()

# Extract unique speaker IDs
speakers = {}
for line in lines:
    _, filepath = line.strip().split()
    speaker_id = filepath.split("/")[0]
    if speaker_id not in speakers:
        speakers[speaker_id] = []
    speakers[speaker_id].append(filepath)

# Select random speakers
selected_speakers = random.sample(list(speakers.keys()), num_speakers)

# Write filtered paths to the output file
with open(output_file, "w") as f_out:
    for speaker_id in selected_speakers:
        for path in speakers[speaker_id]:
            f_out.write(f"{speaker_id}/{path}\n")

print(f"Filtered list written to {output_file} with {num_speakers} speakers.")
