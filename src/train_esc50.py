# src/train_esc50.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
from torch.optim import AdamW 
import torch.nn as nn
from tqdm import tqdm
import os
from sklearn.metrics import f1_score

import config
from esc50_dataset import ESC50Dataset

def train_esc50():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=config.ESC50_NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    train_dataset = ESC50Dataset(mode='train')
    val_dataset = ESC50Dataset(mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    print("Starting training on ESC-50...")
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for spectrograms, labels in progress_bar:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # --- THIS IS THE FIX ---
            # Squeeze the tensor to remove the extra dimension at index 3
            if spectrograms.dim() == 5:
                spectrograms = spectrograms.squeeze(3)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = loss_function(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"train_loss": total_loss / (progress_bar.n + 1)})
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for spectrograms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                spectrograms = spectrograms.to(device)
                
                # Also apply the fix here for validation data
                if spectrograms.dim() == 5:
                    spectrograms = spectrograms.squeeze(3)

                outputs = model(spectrograms)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"Epoch {epoch+1} - Validation Macro F1-Score: {val_f1:.4f}")

    print("Training finished!")
    save_path = 'models/ast_finetuned_esc50.bin'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train_esc50()