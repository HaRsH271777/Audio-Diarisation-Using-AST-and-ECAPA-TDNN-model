# src/visualize_embeddings.py

import torch
import numpy as np
import pandas as pd
import os
import librosa
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from speechbrain.inference.classifiers import EncoderClassifier
import config
from snippet_dataset import SnippetDataset

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the FULL EncoderClassifier
    print("Loading fingerprint model...")
    full_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa_voxceleb"
    )
    
    embedding_model = full_classifier.mods.embedding_model
    embedding_model.load_state_dict(torch.load('models/ecapa_embedding_model_advanced.bin'))
    embedding_model.to(device).eval()

    # 2. Load the snippet dataset
    print("Loading snippet metadata...")
    full_dataset = SnippetDataset(
        metadata_csv='data/snippets_metadata.csv',
        snippets_dir='data/snippets/'
    )
    subset_indices = np.random.choice(len(full_dataset), size=min(300, len(full_dataset)), replace=False)
    
    all_embeddings = []
    all_labels = []

    print(f"Generating embeddings for {len(subset_indices)} snippets...")
    with torch.no_grad():
        for idx in tqdm(subset_indices):
            audio, label = full_dataset[idx]
            audio = audio.unsqueeze(0).to(device)
            
            features = full_classifier.mods.compute_features(audio)
            features = full_classifier.mods.mean_var_norm(features, torch.ones(1).to(device))
            
            emb = embedding_model(features).squeeze().cpu().numpy()
            
            all_embeddings.append(emb)
            all_labels.append(label.item())

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    # 3. Use t-SNE
    print("Running t-SNE... (this can take a moment)")
    # --- THIS IS THE FIX ---
    # The 'n_iter' parameter was renamed to 'max_iter' in newer scikit-learn versions
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # 4. Plot the results
    print("Plotting results...")
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('tab20', config.NUM_CLASSES)
    
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap=cmap)
    
    plt.title("t-SNE Visualization of Bird Call Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    legend_labels = np.unique(all_labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=cmap(i / (config.NUM_CLASSES - 1)),
                        label=f'Species {i}') for i in legend_labels]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("embedding_visualization.png")
    print("Saved visualization to embedding_visualization.png")
    plt.show()

if __name__ == '__main__':
    visualize()