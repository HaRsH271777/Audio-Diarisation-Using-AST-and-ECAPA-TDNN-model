# src/config.py

# -- Project Paths --
# Correct version
DATA_DIR = 'data/rfcx-species-audio-detection/'
TRAIN_TP_CSV = DATA_DIR + 'train_tp.csv'
TFRECORDS_DIR = DATA_DIR + 'tfrecords/train/'

# -- Audio Settings --
SAMPLE_RATE = 48000
CLIP_DURATION = 60  # The original clips are 60s long

# -- Model Input Settings (from Kimi's Plan for AST) --
PATCH_DURATION = 10 # We'll slice the 60s clips into 10s patches
N_MELS = 80        # Number of Mel frequency bins
HOP_LENGTH = 471    # Adjusted to match model's exact expected sequence length
N_FFT = 1024

# -- Training Settings --
NUM_CLASSES = 24 # Number of bird species in the dataset

ESC50_DIR = 'data/ESC-50-master/'
ESC50_META = ESC50_DIR + 'esc50.csv'  # Corrected path
ESC50_AUDIO = ESC50_DIR + 'audio/audio/' # Corrected path
ESC50_NUM_CLASSES = 50
ESC50_DURATION = 5 # Clips are 5s long