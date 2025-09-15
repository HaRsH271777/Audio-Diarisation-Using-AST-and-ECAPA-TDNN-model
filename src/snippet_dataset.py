# src/snippet_dataset.py

import torch
import pandas as pd
import numpy as np
import librosa
import os

import config

class SnippetDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, snippets_dir, duration=3):
        """
        Initializes the Dataset for audio snippets.
        - duration: The fixed length of audio chunks to return, in seconds.
        """
        self.df = pd.read_csv(metadata_csv)
        self.snippets_dir = snippets_dir
        self.duration = duration
        self.target_length = self.duration * config.SAMPLE_RATE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Loads one audio snippet, pads or crops it to a fixed length,
        and returns it with its label.
        """
        row = self.df.iloc[index]
        snippet_path = os.path.join(self.snippets_dir, row['snippet_filename'])
        species_id = row['species_id']

        # Load the audio snippet
        audio, sr = librosa.load(snippet_path, sr=config.SAMPLE_RATE)

        # Pad or crop the audio to the target length
        if len(audio) < self.target_length:
            # Pad with zeros if shorter
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        elif len(audio) > self.target_length:
            # Take a random crop if longer
            start = np.random.randint(0, len(audio) - self.target_length + 1)
            audio = audio[start : start + self.target_length]

        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        label_tensor = torch.tensor(species_id, dtype=torch.long)
        
        return audio_tensor, label_tensor   