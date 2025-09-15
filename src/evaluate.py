# src/evaluate.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification, AutoConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import numpy as np

# Import our custom modules
import config
from dataset import RFCXDataset

def evaluate():
    """
    Main evaluation function.
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

    # --- UPDATED: Load the newest model file ---
    model_path = 'models/ast_finetuned_rfcx_v3.bin'
    print(f"Loading trained weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print("Setting up validation dataloader...")
    val_dataset = RFCXDataset(
        metadata_csv=config.TRAIN_TP_CSV,
        tfrecords_dir=config.TFRECORDS_DIR,
        mode='val'
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=16, 
        shuffle=False,
        num_workers=0
    )

    print("Starting evaluation...")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(val_loader, desc="Evaluating"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            outputs = model(spectrograms)
            
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    print("Calculating metrics...")
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\n--- Validation Results ---")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("------------------------")

if __name__ == '__main__':
    evaluate()
