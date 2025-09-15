# src/train_ecapa.py

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm
import os

# Import our custom modules
import config
from snippet_dataset import SnippetDataset
from pytorch_metric_learning import losses


# --- NEW: Custom Model Class ---
class CustomECAPAModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # These are the modules from the pre-trained SpeechBrain model
        self.compute_features = base_model.mods.compute_features
        self.mean_var_norm = base_model.mods.mean_var_norm
        self.embedding_model = base_model.mods.embedding_model

    def forward(self, x, lengths=None):
        # Pass audio through the feature extraction and normalization
        x = self.compute_features(x)
        x = self.mean_var_norm(x, lengths)
        
        # Get the embeddings
        x = self.embedding_model(x, lengths)
        
        # The ECAPA model returns embeddings with an extra dimension, so we squeeze it
        x = x.squeeze(1)
        
        return x

def train_ecapa_advanced():
    """
    Advanced training function for ECAPA-TDNN using Prototypical Loss.
    """
    # -- 1. Set up --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- 2. Load Pre-trained ECAPA-TDNN Model --
    print("Loading pre-trained ECAPA-TDNN model from SpeechBrain...")
    base_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa_voxceleb"
    )
    
    # -- 3. Create the Full Custom Model --
    full_model = CustomECAPAModel(base_model).to(device)
    for param in full_model.parameters():
        param.requires_grad = True

    # -- 4. Create Datasets and DataLoaders --
    print("Setting up snippet datasets with smart sampler...")
    train_dataset = SnippetDataset(
        metadata_csv='data/snippets_metadata.csv',
        snippets_dir='data/snippets/'
    )
    
    class_counts = train_dataset.df['species_id'].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[species_id] for species_id in train_dataset.df['species_id']]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0
    )

    loss_function = losses.PNPLoss()


    # -- 5. Define Optimizer --
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)

    # -- 6. The Training Loop --
    print("Starting ADVANCED ECAPA-TDNN training...")
    num_epochs = 20

    for epoch in range(num_epochs):
        full_model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for audio_chunks, labels in progress_bar:
            audio_chunks = audio_chunks.to(device)
            labels = labels.to(device)
            
            lengths = torch.ones(audio_chunks.shape[0], device=device)

            optimizer.zero_grad()
            
            embeddings = full_model(audio_chunks, lengths)
            
            loss = loss_function(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

    # -- 7. Save the Fingerprint Model --
    print("Training finished! Saving fingerprint model...")
    save_path = 'models/ecapa_embedding_model_advanced.bin'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(full_model.embedding_model.state_dict(), save_path)
    print(f"Fingerprint model saved to {save_path}")

if __name__ == '__main__':
    train_ecapa_advanced()