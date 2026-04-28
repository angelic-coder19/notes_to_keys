"""
model_utils.py
Notes to Keys — Model Definition and Utilities

Contains:
    - OnsetsAndFrames model architecture (must match training exactly)
    - load_model()       — loads best_model.pth from disk
    - rolls_to_notes()   — converts onset/frame binary matrices to note events
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict


# ============================================================================
# MODEL ARCHITECTURE
# Must be identical to the architecture used during training.
# Do not change any of these values.
# ============================================================================

class AcousticCNN(nn.Module):
    """
    Shared CNN backbone.
    Input:  (batch, 1, 176, 100)
    Output: (batch, 100, feature_dim)
    """
    def __init__(self, n_freq_bins=176, channels=[32, 64, 128], dropout=0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, channels[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),

            # Block 2
            nn.Conv2d(channels[0], channels[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout * 0.5),

            # Block 3
            nn.Conv2d(channels[1], channels[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(dropout),
        )

        # 176 freq bins → 2× MaxPool2d(2,1) → 44 bins remaining
        self.freq_after_pool = n_freq_bins // 4
        self.feature_dim = channels[2] * self.freq_after_pool

    def forward(self, x):
        x = self.conv_layers(x)               # (B, 128, 44, 100)
        batch, c, f, t = x.shape
        x = x.view(batch, c * f, t)           # (B, 5632, 100)
        x = x.permute(0, 2, 1)               # (B, 100, 5632)
        return x


class OnsetsAndFrames(nn.Module):
    """
    Full transcription model.
    Input:  CQT (batch, 1, 176, 100)
    Output: onset_logits (batch, 100, 88)
            frame_logits (batch, 100, 88)
    """
    def __init__(self, cfg: dict):
        super().__init__()

        self.cnn = AcousticCNN(
            n_freq_bins=cfg['n_freq_bins'],
            channels=cfg['cnn_channels'],
            dropout=cfg['dropout'],
        )

        feat_dim = self.cnn.feature_dim
        hidden   = cfg['lstm_hidden']
        layers   = cfg['lstm_layers']
        n_keys   = cfg['n_keys']

        # Onset detector
        self.onset_lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg['dropout'] if layers > 1 else 0.0,
        )
        self.onset_proj = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(hidden * 2, n_keys),
        )

        # Frame detector — conditioned on onset predictions
        self.frame_lstm = nn.LSTM(
            input_size=feat_dim + n_keys,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg['dropout'] if layers > 1 else 0.0,
        )
        self.frame_proj = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(hidden * 2, n_keys),
        )

    def forward(self, cqt):
        features = self.cnn(cqt)                              # (B, T, feat_dim)

        onset_out, _  = self.onset_lstm(features)             # (B, T, hidden*2)
        onset_logits  = self.onset_proj(onset_out)            # (B, T, 88)

        onset_probs   = torch.sigmoid(onset_logits).detach()  # no grad through here
        frame_input   = torch.cat([features, onset_probs], dim=-1)
        frame_out, _  = self.frame_lstm(frame_input)          # (B, T, hidden*2)
        frame_logits  = self.frame_proj(frame_out)            # (B, T, 88)

        return onset_logits, frame_logits


# ============================================================================
# MODEL CONFIG
# These values must match training exactly.
# ============================================================================

MODEL_CFG = {
    'n_freq_bins':   176,
    'n_keys':        88,
    'chunk_frames':  100,
    'cnn_channels':  [32, 64, 128],
    'lstm_hidden':   256,
    'lstm_layers':   2,
    'dropout':       0.3,
    'onset_threshold': 0.5,
    'frame_threshold': 0.5,
}


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(model_path: str, device: torch.device) -> OnsetsAndFrames:
    """
    Load the trained model from a .pth checkpoint file.

    Args:
        model_path: Path to best_model.pth
        device:     torch.device — 'cpu' or 'cuda'

    Returns:
        model in eval mode, ready for inference
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure best_model.pth is in the backend/ folder."
        )

    model = OnsetsAndFrames(MODEL_CFG).to(device)

    checkpoint = torch.load(path, map_location=device)

    # Handle both raw state dict and full checkpoint dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
              f"val_loss {checkpoint.get('val_loss', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights.")

    model.eval()
    print(f"Model ready on {device}.")
    return model


# ============================================================================
# ROLLS TO NOTES
# Converts binary onset/frame matrices into a list of note events.
# ============================================================================

def rolls_to_notes(
    onset_roll: np.ndarray,
    frame_roll: np.ndarray,
    frame_duration_sec: float = 512 / 22050,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    min_note_duration_sec: float = 0.05,
) -> List[Dict]:
    """
    Convert onset and frame probability matrices into discrete note events.

    Args:
        onset_roll:          (T, 88) float — onset probabilities
        frame_roll:          (T, 88) float — frame probabilities
        frame_duration_sec:  seconds per frame (~0.0232s at 22050Hz, hop=512)
        onset_threshold:     probability cutoff for onset detection
        frame_threshold:     probability cutoff for frame activity
        min_note_duration_sec: discard notes shorter than this (removes spurious blips)

    Returns:
        List of dicts, one per note:
        {
            'pitch':      int  — MIDI pitch number (21=A0, 108=C8)
            'key':        int  — piano key index (0=A0, 87=C8)
            'start_time': float — seconds from start of audio
            'end_time':   float — seconds from start of audio
            'duration':   float — note duration in seconds
        }

    How it works:
        For each of the 88 piano keys, we scan the onset roll for rising edges
        (frames where the model predicts a note start). From each onset we then
        scan forward in the frame roll to find where the note ends — either when
        the frame roll drops to 0 or when we reach a new onset for the same key.
        Notes shorter than min_note_duration_sec are discarded.
    """
    T, n_keys = frame_roll.shape
    assert n_keys == 88, f"Expected 88 keys, got {n_keys}"
    assert onset_roll.shape == frame_roll.shape

    # Binarise
    onset_binary = (onset_roll >= onset_threshold).astype(np.uint8)
    frame_binary = (frame_roll >= frame_threshold).astype(np.uint8)

    min_frames = max(1, int(min_note_duration_sec / frame_duration_sec))
    notes = []

    for key in range(n_keys):
        onset_frames = np.where(onset_binary[:, key] == 1)[0]

        # Deduplicate consecutive onset frames — keep only the first frame
        # of each onset cluster (onset_length=3 in preprocessing)
        if len(onset_frames) == 0:
            continue

        deduped_onsets = [onset_frames[0]]
        for f in onset_frames[1:]:
            if f - deduped_onsets[-1] > 3:   # gap larger than onset_length
                deduped_onsets.append(f)

        for onset_frame in deduped_onsets:
            # Find where this note ends by scanning forward in the frame roll
            end_frame = onset_frame
            for f in range(onset_frame, T):
                if frame_binary[f, key] == 1:
                    end_frame = f
                else:
                    break

            # Require at least one frame of sustained activity
            if end_frame <= onset_frame:
                end_frame = onset_frame + 1

            duration_frames = end_frame - onset_frame + 1
            if duration_frames < min_frames:
                continue

            start_sec = onset_frame * frame_duration_sec
            end_sec   = (end_frame + 1) * frame_duration_sec

            notes.append({
                'pitch':      key + 21,        # MIDI pitch (A0=21, C8=108)
                'key':        key,             # piano key index (0–87)
                'start_time': round(start_sec, 4),
                'end_time':   round(end_sec, 4),
                'duration':   round(end_sec - start_sec, 4),
            })

    # Sort by start time, then by pitch
    notes.sort(key=lambda n: (n['start_time'], n['pitch']))
    return notes