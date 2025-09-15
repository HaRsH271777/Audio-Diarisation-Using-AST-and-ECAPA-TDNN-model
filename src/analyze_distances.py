# src/analyze_distances.py

import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from speechbrain.inference.classifiers import EncoderClassifier
from snippet_dataset import SnippetDataset

def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained fingerprint model
    print("Loading fingerprint model...")
    full_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa_voxceleb"
    )
    embedding_model = full_classifier.mods.embedding_model
    embedding_model.load_state_dict(torch.load('models/ecapa_embedding_model_advanced.bin'))
    embedding_model.to(device).eval()

    # 2. Load a subset of the snippet dataset
    print("Loading snippet data...")
    full_dataset = SnippetDataset(
        metadata_csv='data/snippets_metadata.csv',
        snippets_dir='data/snippets/'
    )
    subset_indices = np.random.choice(len(full_dataset), size=min(100, len(full_dataset)), replace=False)
    
    all_embeddings = []
    with torch.no_grad():
        for idx in tqdm(subset_indices, desc="Generating Embeddings"):
            audio, _ = full_dataset[idx]
            audio = audio.unsqueeze(0).to(device)
            
            features = full_classifier.mods.compute_features(audio)
            features = full_classifier.mods.mean_var_norm(features, torch.ones(1).to(device))
            
            emb = embedding_model(features).squeeze().cpu().numpy()
            all_embeddings.append(emb)

    all_embeddings = np.array(all_embeddings)

    # 3. Calculate pairwise distances between all embeddings
    print("\nCalculating pairwise distances...")
    # pdist calculates the condensed distance matrix
    distances = pdist(all_embeddings, metric='euclidean')
    
    # 4. Print the statistics
    print("\n--- Embedding Distance Statistics ---")
    print(f"Minimum distance: {np.min(distances):.2f}")
    print(f"Mean distance:    {np.mean(distances):.2f}")
    print(f"Median distance:  {np.median(distances):.2f}")
    print(f"Maximum distance: {np.max(distances):.2f}")
    print("-----------------------------------")
    print("\nRecommendation: Try a `distance_threshold` in diarize.py that is slightly less than the mean/median distance.")


if __name__ == '__main__':
    analyze()