# src/prepare_snippet_data.py

import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Import our settings
import config

def prepare_snippets():
    """
    Finds all labeled bird calls, extracts them as short audio clips,
    and saves them to a new directory.
    """
    # --- Configuration ---
    # We will use the raw .flac files for this
    INPUT_AUDIO_DIR = 'data/rfcx-species-audio-detection/train/'
    OUTPUT_SNIPPET_DIR = 'data/snippets/'
    METADATA_FILE = config.TRAIN_TP_CSV
    
    MIN_DURATION = 1.0  # seconds
    MAX_DURATION = 15.0 # seconds
    
    # --- Main Logic ---
    print("Starting snippet preparation...")
    # 1. Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_SNIPPET_DIR, exist_ok=True)
    
    # 2. Load the metadata with all the labeled calls
    df = pd.read_csv(METADATA_FILE)
    
    new_metadata = []
    
    # 3. Iterate through each labeled call
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Snippets"):
        recording_id = row['recording_id']
        species_id = row['species_id']
        t_min, t_max = row['t_min'], row['t_max']
        
        duration = t_max - t_min
        
        # 4. Filter out calls that are too short or too long
        if MIN_DURATION <= duration <= MAX_DURATION:
            input_path = os.path.join(INPUT_AUDIO_DIR, f"{recording_id}.flac")
            
            # 5. Use librosa to load only the specific audio slice
            # This is efficient as it doesn't load the full 60s clip
            try:
                snippet, _ = librosa.load(
                    input_path,
                    sr=config.SAMPLE_RATE,
                    offset=t_min,
                    duration=duration
                )
            except Exception as e:
                print(f"Warning: Could not load {input_path}. Error: {e}")
                continue

            # 6. Save the new snippet as a .wav file
            output_filename = f"{recording_id}_sp{species_id}_{index}.wav"
            output_path = os.path.join(OUTPUT_SNIPPET_DIR, output_filename)
            sf.write(output_path, snippet, config.SAMPLE_RATE)
            
            # 7. Store the metadata for our new snippet file
            new_metadata.append({
                'snippet_filename': output_filename,
                'species_id': species_id,
                'recording_id': recording_id
            })

    # 8. Save the new metadata to a CSV file
    new_metadata_df = pd.DataFrame(new_metadata)
    new_metadata_df.to_csv('data/snippets_metadata.csv', index=False)
    
    print(f"\nSnippet preparation complete!")
    print(f"Saved {len(new_metadata_df)} snippets to '{OUTPUT_SNIPPET_DIR}'")
    print(f"Saved new metadata to 'data/snippets_metadata.csv'")


if __name__ == '__main__':
    prepare_snippets()  