# src/diarize.py

import torch
import librosa
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.lobes.features import Fbank
import pandas as pd
from tqdm import tqdm

import config
from snippet_dataset import SnippetDataset

def load_fingerprint_model_and_database(device):
    """
    Loads the fingerprint model and creates a reference database of average
    embeddings for each species.
    """
    print("Loading fingerprint model...")
    embedding_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa_voxceleb"
    ).mods.embedding_model
    embedding_model.load_state_dict(torch.load('models/ecapa_embedding_model_advanced.bin'))
    embedding_model.to(device).eval()

    fbank_extractor = Fbank(
        sample_rate=config.SAMPLE_RATE, n_mels=80, win_length=25,
        hop_length=10, n_fft=2048
    ).to(device)

    # --- NEW: Build the Reference Fingerprint Database ---
    print("Building reference fingerprint database...")
    snippet_dataset = SnippetDataset(
        metadata_csv='data/snippets_metadata.csv',
        snippets_dir='data/snippets/'
    )
    
    species_embeddings = {i: [] for i in range(config.NUM_CLASSES)}
    with torch.no_grad():
        for audio, label in tqdm(snippet_dataset, desc="Generating reference fingerprints"):
            audio = audio.unsqueeze(0).to(device)
            features = fbank_extractor(audio)
            lengths = torch.ones(features.shape[0], device=device)
            emb = embedding_model(features, lengths=lengths).squeeze().cpu().numpy()
            species_embeddings[label.item()].append(emb)

    reference_database = {}
    for species_id, embs in species_embeddings.items():
        if embs:
            reference_database[species_id] = np.mean(embs, axis=0)
    print("Reference database complete.")
    
    return embedding_model, fbank_extractor, reference_database

def diarize_audio(audio_path, embedding_model, fbank_extractor, reference_database, device):
    """
    Performs full diarization using only the embedding model and fingerprint matching.
    """
    print(f"\n--- Starting Diarization for {os.path.basename(audio_path)} ---")
    
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    
    print("Step 1: Generating fingerprints for the long audio file...")
    window_size = int(1.5 * sr); step_size = int(0.75 * sr)
    all_embeddings = []; timestamps = []
    
    for start in range(0, len(audio) - window_size, step_size):
        end = start + window_size
        window = torch.from_numpy(audio[start:end]).unsqueeze(0).to(device)
        with torch.no_grad():
            features = fbank_extractor(window)
            lengths = torch.ones(features.shape[0], device=device)
            emb = embedding_model(features, lengths=lengths).squeeze().cpu().numpy()
            all_embeddings.append(emb)
            timestamps.append(start / sr)

    all_embeddings = np.array(all_embeddings)
    
    print("Step 2: Clustering fingerprints to find distinct calls...")
    # This threshold is the key parameter to tune.
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=345.0).fit(all_embeddings)
    labels = clustering.labels_
    num_calls = labels.max() + 1
    print(f"Found {num_calls} potential bird calls.")
    
    print("Step 3: Matching each call to the reference database...")
    results = []
    # Convert reference database to an array for easy distance calculation
    ref_species_ids = list(reference_database.keys())
    ref_embeddings = np.array(list(reference_database.values()))

    for i in range(num_calls):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < 2:
            continue
            
        start_time = timestamps[cluster_indices[0]]
        end_time = timestamps[cluster_indices[-1]] + (window_size / sr)
        
        # Calculate the average fingerprint for this detected call
        cluster_embeddings = all_embeddings[cluster_indices]
        mean_cluster_embedding = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        
        # Find the closest species in our reference database
        distances = cdist(mean_cluster_embedding, ref_embeddings, metric='euclidean')
        closest_species_index = np.argmin(distances)
        predicted_species_id = ref_species_ids[closest_species_index]
        
        results.append({
            "species_id": predicted_species_id,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2)
        })
        
    return results

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model, fbank_extractor, reference_database = load_fingerprint_model_and_database(device)
    
    test_audio_file = 'data/rfcx-species-audio-detection/train/0a19197c4.flac'
    detected_calls = diarize_audio(test_audio_file, embedding_model, fbank_extractor, reference_database, device)
    
    print("\n--- Diarization Complete ---")
    if detected_calls:
        for call in detected_calls:
            print(f"Detected Species ID: {call['species_id']}, Start: {call['start_time']}s, End: {call['end_time']}s")
    else:
        print("No distinct calls were detected in this file.")