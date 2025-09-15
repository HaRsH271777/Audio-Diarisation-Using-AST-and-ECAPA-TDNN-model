# src/evaluate_diarization.py

import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from sklearn.model_selection import train_test_split
import os
import torch

# --- UPDATED: Import the new functions from our final diarize.py ---
from diarize import load_fingerprint_model_and_database, diarize_audio
import config

def run_final_evaluation():
    # 1. Load the fingerprint model and reference database
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model, fbank_extractor, reference_database = load_fingerprint_model_and_database(device)

    # 2. Get the validation set of recordings
    df = pd.read_csv(config.TRAIN_TP_CSV)
    all_recording_ids = df['recording_id'].unique()
    _, val_ids = train_test_split(all_recording_ids, test_size=0.2, random_state=42)
    
    val_ids_subset = val_ids[:5] 
    print(f"\nRunning final evaluation on {len(val_ids_subset)} files...")

    # 3. Initialize the DER metric
    metric = DiarizationErrorRate()
    
    # 4. Loop through validation files, get predictions, and compare to ground truth
    for rec_id in val_ids_subset:
        print(f"\nProcessing recording: {rec_id}")
        audio_path = os.path.join('data/rfcx-species-audio-detection/train', f"{rec_id}.flac")

        # Get the "ground truth" labels from the CSV
        truth_rows = df[df.recording_id == rec_id]
        reference = Annotation()
        for _, row in truth_rows.iterrows():
            reference[Segment(row['t_min'], row['t_max'])] = str(row['species_id'])

        # Get our model's predicted labels using the new diarization function
        predicted_calls = diarize_audio(audio_path, embedding_model, fbank_extractor, reference_database, device)
        hypothesis = Annotation()
        for call in predicted_calls:
            hypothesis[Segment(call['start_time'], call['end_time'])] = str(call['species_id'])
            
        # Calculate the DER for this one file
        der_result = metric(reference, hypothesis)
        print(f"DER for this file: {der_result * 100:.2f}%")

    # 5. Get the final, total DER across all files
    total_der = abs(metric)
    print("\n--- Final Diarization Results ---")
    print(f"Total Diarization Error Rate (DER): {total_der * 100:.2f}%")
    print("---------------------------------")


if __name__ == '__main__':
    run_final_evaluation()