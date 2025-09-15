# src/esc50_dataset.py

import torch
import pandas as pd
import numpy as np
import librosa
import os

import config

class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        df = pd.read_csv(config.ESC50_META)
        
        if self.mode == 'train':
            self.df = df[df['fold'] != 5]
        elif self.mode == 'val':
            self.df = df[df['fold'] == 5]
            
        print(f"Created {self.mode} dataset with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio_path = os.path.join(config.ESC50_AUDIO, row['filename'])
        label = row['target']
        
        audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, duration=config.ESC50_DURATION)
        
        target_length = config.ESC50_DURATION * config.SAMPLE_RATE
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
        )
        pcen_spec = librosa.pcen(mel_spec, sr=config.SAMPLE_RATE)
        
        # Reshape and transpose to match the expected input shape for AST model [channels, height, width]
        pcen_spec_tensor = torch.from_numpy(pcen_spec.astype(np.float32))
        pcen_spec_tensor = pcen_spec_tensor.unsqueeze(0)  # Add channel dimension
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return pcen_spec_tensor, label_tensor