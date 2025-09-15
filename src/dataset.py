# src/dataset.py

import torch
import tensorflow as tf
import pandas as pd
import numpy as np
import librosa
from audiomentations import Compose
from audiomentations.augmentations.frequency_mask import FrequencyMask
from audiomentations.augmentations.time_mask import TimeMask
from tqdm import tqdm
from sklearn.model_selection import train_test_split # New import
import random

import config

class RFCXDataset(torch.utils.data.Dataset):
    # --- MODIFIED: Added mode, test_size, and random_state parameters ---
    def __init__(self, metadata_csv, tfrecords_dir, mode='train', test_size=0.2, random_state=42):
        """
        Initializes the Dataset object.
        - mode: 'train' or 'val'. Determines which data split to use.
        """
        self.df = pd.read_csv(metadata_csv)
        self.mode = mode
        
        # --- NEW: Splitting logic ---
        # 1. Get all unique recording IDs
        all_recording_ids = self.df['recording_id'].unique()
        
        # 2. Split the IDs into training and validation sets
        train_ids, val_ids = train_test_split(
            all_recording_ids,
            test_size=test_size,
            random_state=random_state
        )
        
        # 3. Select the IDs for the current mode ('train' or 'val')
        if mode == 'train':
            self.unique_recording_ids = list(train_ids)
            print(f"Created training dataset with {len(self.unique_recording_ids)} recordings.")
        elif mode == 'val':
            self.unique_recording_ids = list(val_ids)
            print(f"Created validation dataset with {len(self.unique_recording_ids)} recordings.")
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'val'.")

        # The rest of the __init__ method is the same
        tfrec_filepaths = tf.io.gfile.glob(tfrecords_dir + '*.tfrec')
        
        print("Pre-loading audio data into memory... This might take a moment.")
        self.audio_data = self._load_all_audio(tfrec_filepaths)
        print("Audio data pre-loading complete.")
        
        patch_samples = config.PATCH_DURATION * config.SAMPLE_RATE
        time_steps = int(np.floor((patch_samples - config.N_FFT) / config.HOP_LENGTH)) + 1

        self.augment = Compose([
            FrequencyMask(min_frequency_band=0.01, max_frequency_band=27/config.N_MELS, p=1.0),
            FrequencyMask(min_frequency_band=0.01, max_frequency_band=27/config.N_MELS, p=1.0),
            TimeMask(min_band_part=0.0, max_band_part=40/time_steps, p=1.0),
            TimeMask(min_band_part=0.0, max_band_part=40/time_steps, p=1.0),
        ])
        
    def _load_all_audio(self, tfrec_filepaths):
        audio_data = {}
        if not tfrec_filepaths: 
            return audio_data
        
        dataset = tf.data.TFRecordDataset(tfrec_filepaths).map(self.parse_tfrecord)
        
        # --- MODIFIED: Only load audio files that are in the current split ---
        print("Filtering audio files for the current split...")
        ids_in_split = set(self.unique_recording_ids)
        
        for wav, rec_id_tensor in tqdm(dataset, desc="Loading Audio Files"):
            rec_id = rec_id_tensor.numpy().decode('utf-8')
            if rec_id in ids_in_split: # Only load if the ID is in our train/val set
                audio_data[rec_id] = wav.numpy()
        return audio_data

    # --- No changes to the methods below this line ---

    def __len__(self):
        return len(self.unique_recording_ids)

    def parse_tfrecord(self, example_proto):
        feature_description = {
            "recording_id": tf.io.FixedLenFeature([], tf.string),
            "audio_wav": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        audio_wav = tf.io.decode_raw(example['audio_wav'], tf.int16)
        audio_wav = tf.cast(audio_wav, tf.float32) / 32768.0
        return audio_wav, example['recording_id']

    def __getitem__(self, index):
        recording_id = self.unique_recording_ids[index]
        audio_wav = self.audio_data.get(recording_id)
        
        if audio_wav is None:
            raise ValueError(f"Recording ID {recording_id} not found in pre-loaded data.")

        patch_samples = config.PATCH_DURATION * config.SAMPLE_RATE
        max_start = len(audio_wav) - patch_samples
        if max_start <= 0:
            max_start = 1
        start_sample = np.random.randint(0, max_start)
        audio_patch = audio_wav[start_sample : start_sample + patch_samples]

        slice_start_sec = start_sample / config.SAMPLE_RATE
        slice_end_sec = (start_sample + patch_samples) / config.SAMPLE_RATE
        relevant_rows = self.df[self.df.recording_id == recording_id]
        calls_in_slice = relevant_rows[
            (relevant_rows.t_min < slice_end_sec) & (relevant_rows.t_max > slice_start_sec)
        ]
        label = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        species_in_slice = calls_in_slice.species_id.unique()
        for species_id in species_in_slice:
            label[species_id] = 1.0

        mel_spec = librosa.feature.melspectrogram(
            y=audio_patch, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
        )
        
        # In validation mode, we shouldn't use augmentations
        if self.mode == 'train' and random.random() < 0.5:
             augmented_spec = self.augment(samples=mel_spec, sample_rate=config.SAMPLE_RATE)
        else:
             augmented_spec = mel_spec

        pcen_spec = librosa.pcen(augmented_spec, sr=config.SAMPLE_RATE)

        pcen_spec_tensor = torch.from_numpy(pcen_spec.astype(np.float32))
        label_tensor = torch.from_numpy(label)
        
        return pcen_spec_tensor, label_tensor   