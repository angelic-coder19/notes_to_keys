"""
inference.py
Notes to Keys — Inference Pipeline

The single entry point for transcribing audio.
Takes a path to an audio file, returns a structured transcription result.

Usage (standalone test):
    python inference.py path/to/audio.wav

Usage (from Flask):
    from inference import TranscriptionPipeline
    pipeline = TranscriptionPipeline('best_model.pth')
    result = pipeline.transcribe('audio.wav')
"""

import numpy as np
import librosa
import torch
import time
import sys
from pathlib import Path
from typing import Dict, Any

from model_utils import load_model, rolls_to_notes, MODEL_CFG


# ============================================================================
# PREPROCESSING CONFIG
# Must match main2.py exactly.
# ============================================================================

PREPROC_CFG = {
    'sample_rate':    22050,
    'hop_length':     512,
    'n_bins':         176,
    'bins_per_octave':24,
    'fmin':           27.5,      # A0 — lowest piano key
    'global_mean':    -40.0,
    'global_std':     20.0,
    'chunk_length':   100,       # frames per chunk
}

# Seconds per frame — derived from sample_rate and hop_length
FRAME_DURATION_SEC = PREPROC_CFG['hop_length'] / PREPROC_CFG['sample_rate']

# Maximum audio duration to accept (seconds)
# Prevents memory issues with very long files
MAX_AUDIO_DURATION_SEC = 600   # 10 minutes


# ============================================================================
# TRANSCRIPTION PIPELINE
# ============================================================================

class TranscriptionPipeline:
    """
    Loads the model once at startup and exposes a single transcribe() method.
    Designed to be instantiated once when the Flask app starts, then reused
    for every request.
    """

    def __init__(self, model_path: str = 'best_model.pth'):
        """
        Args:
            model_path: Path to best_model.pth
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Inference device: {self.device}")

        self.model = load_model(model_path, self.device)
        self.model_path = model_path

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD
    # ──────────────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Full inference pipeline: audio file → transcription result.

        Args:
            audio_path: Path to a .wav or .mp3 audio file

        Returns:
            {
                'notes':         list of note events (see rolls_to_notes)
                'onset_roll':    list of lists — (T, 88) onset probabilities
                'frame_roll':    list of lists — (T, 88) frame probabilities
                'duration_sec':  float — total audio duration
                'n_frames':      int   — total number of frames
                'frame_duration_sec': float — seconds per frame
                'n_notes':       int   — number of detected notes
                'inference_time_sec': float — how long inference took
            }

        Raises:
            FileNotFoundError: if audio file does not exist
            ValueError:        if audio is too long or has no content
        """
        t_start = time.time()

        # 1. Validate input
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 2. Load and validate audio
        signal = self._load_audio(path)

        # 3. Compute CQT
        cqt = self._compute_cqt(signal)

        # 4. Split into overlapping chunks
        chunks = self._make_chunks(cqt)

        # 5. Run model forward pass
        onset_roll, frame_roll = self._run_model(chunks)

        # 6. Convert to note events
        notes = rolls_to_notes(
            onset_roll,
            frame_roll,
            frame_duration_sec=FRAME_DURATION_SEC,
            onset_threshold=MODEL_CFG['onset_threshold'],
            frame_threshold=MODEL_CFG['frame_threshold'],
        )

        t_elapsed = time.time() - t_start
        n_frames  = onset_roll.shape[0]

        return {
            'notes':              notes,
            'onset_roll':         onset_roll.tolist(),
            'frame_roll':         frame_roll.tolist(),
            'duration_sec':       round(n_frames * FRAME_DURATION_SEC, 3),
            'n_frames':           n_frames,
            'frame_duration_sec': round(FRAME_DURATION_SEC, 6),
            'n_notes':            len(notes),
            'inference_time_sec': round(t_elapsed, 3),
        }

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _load_audio(self, path: Path) -> np.ndarray:
        """
        Load audio file, validate duration, return mono float32 signal.

        Librosa handles: .wav, .mp3, .flac, .ogg and more.
        Always resamples to PREPROC_CFG['sample_rate'].
        """
        print(f"Loading audio: {path.name}")

        signal, sr = librosa.load(
            str(path),
            sr=PREPROC_CFG['sample_rate'],
            mono=True,
        )

        duration_sec = len(signal) / PREPROC_CFG['sample_rate']
        print(f"  Duration: {duration_sec:.1f}s  Samples: {len(signal):,}  SR: {sr}Hz")

        if duration_sec > MAX_AUDIO_DURATION_SEC:
            raise ValueError(
                f"Audio is {duration_sec:.0f}s long. "
                f"Maximum supported duration is {MAX_AUDIO_DURATION_SEC}s."
            )

        if len(signal) < PREPROC_CFG['hop_length'] * PREPROC_CFG['chunk_length']:
            raise ValueError(
                f"Audio is too short ({duration_sec:.2f}s). "
                f"Minimum length is approximately "
                f"{PREPROC_CFG['hop_length'] * PREPROC_CFG['chunk_length'] / PREPROC_CFG['sample_rate']:.2f}s."
            )

        return signal.astype(np.float32)

    def _compute_cqt(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute normalised CQT spectrogram from raw audio signal.

        Matches preprocessing in main2.py exactly:
            - CQT magnitude → dB → standardise with global mean/std
            - Output shape: (176, T)
        """
        print("  Computing CQT...")

        cqt_complex = librosa.cqt(
            signal,
            sr=PREPROC_CFG['sample_rate'],
            fmin=PREPROC_CFG['fmin'],
            n_bins=PREPROC_CFG['n_bins'],
            bins_per_octave=PREPROC_CFG['bins_per_octave'],
            hop_length=PREPROC_CFG['hop_length'],
        )

        # Magnitude → dB
        cqt_mag = np.abs(cqt_complex)
        cqt_db  = librosa.amplitude_to_db(cqt_mag, ref=np.max)

        # Standardise — same values used in training
        cqt_norm = (cqt_db - PREPROC_CFG['global_mean']) / PREPROC_CFG['global_std']

        print(f"  CQT shape: {cqt_norm.shape}  "
              f"range: [{cqt_norm.min():.2f}, {cqt_norm.max():.2f}]")

        return cqt_norm.astype(np.float32)   # (176, T)

    def _make_chunks(self, cqt: np.ndarray) -> np.ndarray:
        """
        Split CQT into non-overlapping chunks of chunk_length frames.
        The final partial chunk is padded with zeros.

        Returns:
            np.ndarray of shape (N_chunks, 1, 176, chunk_length)
        """
        chunk_len  = PREPROC_CFG['chunk_length']
        n_freq, T  = cqt.shape

        # Pad to a multiple of chunk_len
        remainder = T % chunk_len
        if remainder != 0:
            pad_width = chunk_len - remainder
            cqt = np.pad(cqt, ((0, 0), (0, pad_width)), mode='constant')

        n_frames    = cqt.shape[1]
        n_chunks    = n_frames // chunk_len

        # Reshape into chunks and add channel dimension
        # (176, N*100) → (N, 176, 100) → (N, 1, 176, 100)
        chunks = cqt.T.reshape(n_chunks, chunk_len, n_freq)   # (N, 100, 176)
        chunks = chunks.transpose(0, 2, 1)                     # (N, 176, 100)
        chunks = chunks[:, np.newaxis, :, :]                   # (N, 1, 176, 100)

        print(f"  Chunks: {n_chunks}  shape per chunk: (1, {n_freq}, {chunk_len})")
        return chunks.astype(np.float32)

    def _run_model(self, chunks: np.ndarray) -> tuple:
        """
        Run the model forward pass on all chunks.
        Processes in batches of 16 to avoid memory issues on long audio.

        Args:
            chunks: (N, 1, 176, 100) float32 numpy array

        Returns:
            onset_roll: (T_total, 88) float32 — onset probabilities
            frame_roll: (T_total, 88) float32 — frame probabilities
        """
        batch_size = 16
        n_chunks   = len(chunks)
        print(f"  Running model on {n_chunks} chunks (batch_size={batch_size})...")

        all_onset = []
        all_frame = []

        with torch.no_grad():
            for i in range(0, n_chunks, batch_size):
                batch = torch.from_numpy(chunks[i:i + batch_size]).to(self.device)

                onset_logits, frame_logits = self.model(batch)

                onset_probs = torch.sigmoid(onset_logits).cpu().numpy()  # (B, 100, 88)
                frame_probs = torch.sigmoid(frame_logits).cpu().numpy()  # (B, 100, 88)

                all_onset.append(onset_probs)
                all_frame.append(frame_probs)

        # Concatenate all chunks → single timeline
        onset_roll = np.concatenate(all_onset, axis=0)   # (N, 100, 88)
        frame_roll = np.concatenate(all_frame, axis=0)   # (N, 100, 88)

        # Flatten chunk dimension: (N, 100, 88) → (N*100, 88)
        onset_roll = onset_roll.reshape(-1, 88)
        frame_roll = frame_roll.reshape(-1, 88)

        print(f"  Output shape: {onset_roll.shape}  "
              f"Onset active: {(onset_roll > 0.5).mean()*100:.1f}%  "
              f"Frame active: {(frame_roll > 0.5).mean()*100:.1f}%")

        return onset_roll, frame_roll


# ============================================================================
# STANDALONE TEST
# Run: python inference.py path/to/audio.wav
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio.wav>")
        print()
        print("Tests the inference pipeline end-to-end on a single audio file.")
        print("Make sure best_model.pth is in the same directory.")
        sys.exit(1)

    audio_path = sys.argv[1]

    print()
    print("=" * 60)
    print("  Notes to Keys — Inference Test")
    print("=" * 60)

    pipeline = TranscriptionPipeline('best_model.pth')

    print()
    print(f"Transcribing: {audio_path}")
    print("-" * 60)

    result = pipeline.transcribe(audio_path)

    print()
    print("=" * 60)
    print("  RESULT SUMMARY")
    print("=" * 60)
    print(f"  Audio duration:   {result['duration_sec']:.1f}s")
    print(f"  Total frames:     {result['n_frames']:,}")
    print(f"  Notes detected:   {result['n_notes']}")
    print(f"  Inference time:   {result['inference_time_sec']:.2f}s")
    print()

    if result['n_notes'] > 0:
        print("  First 10 notes:")
        print(f"  {'Key':>4}  {'Pitch':>6}  {'Start':>8}  {'End':>8}  {'Duration':>9}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}")
        for note in result['notes'][:10]:
            print(f"  {note['key']:>4}  "
                  f"{note['pitch']:>6}  "
                  f"{note['start_time']:>8.3f}s  "
                  f"{note['end_time']:>8.3f}s  "
                  f"{note['duration']:>8.3f}s")
    else:
        print("  No notes detected. Check that the audio contains piano.")

    print()
    print("  onset_roll shape:", len(result['onset_roll']),
          "×", len(result['onset_roll'][0]))
    print("  frame_roll shape:", len(result['frame_roll']),
          "×", len(result['frame_roll'][0]))
    print()
    print("  ✅ Pipeline working correctly." if result['n_notes'] > 0
          else "  ⚠️  No notes detected — verify audio file contains piano.")
    print("=" * 60)