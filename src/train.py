# src/train.py

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForAudioClassification, AutoConfig
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

# Import our custom modules
import config
from dataset import RFCXDataset
from focal_loss import FocalLoss # Import our new loss

def train():
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    model_config = AutoConfig.from_pretrained(model_id)
    model_config.attention_dropout = 0.2
    model_config.dropout = 0.2
    model_config.num_labels = config.NUM_CLASSES
    
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # WeightedRandomSampler Logic (no changes here)
    print("Setting up dataset with WeightedRandomSampler...")
    full_df = pd.read_csv(config.TRAIN_TP_CSV)
    train_dataset = RFCXDataset(
        metadata_csv=config.TRAIN_TP_CSV,
        tfrecords_dir=config.TFRECORDS_DIR,
        mode='train'
    )
    class_counts = full_df['species_id'].value_counts().to_dict()
    
    sample_weights = []
    for rec_id in train_dataset.unique_recording_ids:
        species_in_rec = train_dataset.df[train_dataset.df.recording_id == rec_id]['species_id'].unique()
        rec_weight = 0.0
        if len(species_in_rec) > 0:
            for s_id in species_in_rec:
                rec_weight += 1.0 / class_counts.get(s_id, float('inf'))
            rec_weight /= len(species_in_rec)
        sample_weights.append(rec_weight)
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=0
    )

    # loss_function = nn.BCEWithLogitsLoss()
    loss_function = FocalLoss() # Use the more advanced Focal Loss
    
    # --- NEW: Add Weight Decay to the optimizer to reduce overfitting ---
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=10, pct_start=0.1
    )
    
    print("Starting training with regularization...")
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for spectrograms, labels in progress_bar:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = loss_function(outputs.logits, labels)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        
    print("Training finished!")
    
    save_path = 'models/ast_finetuned_rfcx_v3.bin' # Saving as a new version
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully.")

if __name__ == '__main__':
    train()