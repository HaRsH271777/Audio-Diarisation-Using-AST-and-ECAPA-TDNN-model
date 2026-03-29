# Audio Diarisation Using AST and ECAPA-TDNN

Audio diarisation pipeline that combines:
- **AST (Audio Spectrogram Transformer)** for audio event / species classification experiments (notebook-driven), and
- **ECAPA‑TDNN embeddings** (via SpeechBrain) for **fingerprint-style matching + clustering** to segment a long recording into distinct “calls” and assign each segment to the closest reference class.

The repository is primarily notebook-based, with reusable training/diarisation scripts in `src/`.

---

## Repository structure

- `notebooks/`
  - `01_eda.ipynb` — exploratory analysis / experimentation
- `src/`
  - `train_ecapa.py` — trains an ECAPA embedding model (fingerprint model) and saves weights
  - `diarize.py` — runs clustering-based diarisation on a long audio file and matches clusters to a reference embedding database
  - `dataset.py`, `snippet_dataset.py`, `prepare_snippet_data.py` — dataset & snippet preparation utilities
  - `evaluate.py`, `evaluate_diarization.py` — evaluation helpers
  - `visualize_embeddings.py`, `analyze_distances.py` — analysis/visualization utilities
  - `config.py` — shared configuration (sample rate, number of classes, etc.)
- `data/` — expected location for datasets, metadata, and generated snippets
- `requirements.txt` — Python dependencies

---

## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include: `torch`, `torchaudio`, `transformers`, `librosa`, `speechbrain`, `scikit-learn`, `audiomentations`, `pytorch-metric-learning`.

---

## Data expectations

The ECAPA training + diarisation scripts assume you have snippet metadata and snippet audio available at:

- `data/snippets_metadata.csv`
- `data/snippets/`

The diarisation example in `src/diarize.py` also references an example long audio file path:

- `data/rfcx-species-audio-detection/train/0a19197c4.flac`

If your dataset lives elsewhere, update the paths in the scripts or adapt them via `src/config.py` / your own wrapper script.

---

## Train the ECAPA fingerprint model

`src/train_ecapa.py` fine-tunes an ECAPA‑TDNN embedding network (SpeechBrain backbone) using a metric-learning loss, then saves the learned embedding model weights to:

- `models/ecapa_embedding_model_advanced.bin`

Run:

```bash
python src/train_ecapa.py
```

Notes:
- Training uses a **WeightedRandomSampler** to reduce class imbalance effects.
- Default epochs are set inside the script (`num_epochs = 20`).
- If you have a GPU, it will automatically use CUDA when available.

---

## Run diarisation on a long audio file

`src/diarize.py` performs diarisation in 3 steps:
1. Sliding-window embedding extraction
2. Agglomerative clustering to find distinct calls
3. Matching each call (cluster) to the closest class in a **reference fingerprint database** (mean embeddings computed from your snippet dataset)

Run:

```bash
python src/diarize.py
```

Output is printed as a list of detected segments:

- `species_id`
- `start_time` (seconds)
- `end_time` (seconds)

Important knobs:
- The clustering sensitivity is controlled by `distance_threshold` in `AgglomerativeClustering(...)` inside `src/diarize.py`. You’ll likely need to tune it for your dataset.

---

## Notebooks

Start Jupyter and open the notebook(s):

```bash
jupyter notebook
```

Then open:

- `notebooks/01_eda.ipynb`

---

## Reproducibility tips

- Keep `src/config.py` as the single source of truth for sample rate / class counts.
- Version your snippet metadata (`data/snippets_metadata.csv`) if you regenerate snippets.
- Save trained models under `models/` (the training script already does this).

---

## License

No license file is currently included. If you plan to share or reuse this work, consider adding a `LICENSE` (MIT/Apache-2.0/GPL, etc.).

---

## Acknowledgements

- [SpeechBrain](https://github.com/speechbrain/speechbrain) for ECAPA‑TDNN inference and pretrained checkpoints.
- AST / Transformers ecosystem for audio transformer-based modeling.
