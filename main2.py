"""
Notes to Keys - Audio Preprocessing Pipeline
Refactored for production use with incremental processing, logging, and validation.

Usage:
    # Quick testing (loads into RAM)
    python main.py <dataset_path> --mode test --max-files 5
    
    # Full preprocessing (saves to disk)
    python main.py <dataset_path> --mode incremental --output processed_chunks
    
    # Resume interrupted preprocessing
    python main.py <dataset_path> --mode incremental --output processed_chunks --resume
"""

import numpy as np
import librosa
import pretty_midi
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PreprocessingConfig:
    """Configuration for audio preprocessing."""
    # Audio parameters
    sample_rate: int = 22050
    hop_length: int = 512
    
    # CQT parameters
    n_bins: int = 176  # 88 keys × 2 bins per semitone
    bins_per_octave: int = 24
    fmin: float = 27.5  # A0
    
    # Normalization (estimate from dataset)
    global_mean: float = -40.0
    global_std: float = 20.0
    
    # Chunking parameters
    chunk_length: int = 100
    hop_length_chunks: int = 50
    
    # MIDI parameters
    onset_length: int = 3  # frames to mark as onset
    
    def save(self, path: str):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON."""
        with open(path, 'r') as f:
            return cls(**json.load(f))

# Default configuration
CONFIG = PreprocessingConfig()

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: Optional[Path] = None, verbose: bool = True):
    """Setup logging to file and console."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if output directory provided)
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        log_file = output_dir / f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    return logger

logger = logging.getLogger(__name__)

# ============================================================================
# VALIDATION
# ============================================================================

def validate_cqt_shape(cqt: np.ndarray, expected_bins: int = 176) -> bool:
    """Validate CQT spectrogram shape."""
    if cqt.ndim != 3:
        logger.error(f"CQT should be 3D, got {cqt.ndim}D")
        return False
    if cqt.shape[0] != 1:
        logger.error(f"CQT should have 1 channel, got {cqt.shape[0]}")
        return False
    if cqt.shape[1] != expected_bins:
        logger.error(f"CQT should have {expected_bins} bins, got {cqt.shape[1]}")
        return False
    return True

def validate_piano_roll_shape(roll: np.ndarray, expected_keys: int = 88) -> bool:
    """Validate piano roll shape."""
    if roll.ndim != 2:
        logger.error(f"Piano roll should be 2D, got {roll.ndim}D")
        return False
    if roll.shape[1] != expected_keys:
        logger.error(f"Piano roll should have {expected_keys} keys, got {roll.shape[1]}")
        return False
    return True

def validate_data_ranges(onset: np.ndarray, frame: np.ndarray) -> bool:
    """Validate that onset and frame values are in valid range."""
    if not np.all((onset >= 0) & (onset <= 1)):
        logger.error(f"Onset values out of range [0,1]: min={onset.min()}, max={onset.max()}")
        return False
    if not np.all((frame >= 0) & (frame <= 1)):
        logger.error(f"Frame values out of range [0,1]: min={frame.min()}, max={frame.max()}")
        return False
    return True

def validate_processed_chunk(cqt: np.ndarray, onset: np.ndarray, frame: np.ndarray) -> bool:
    """Run all validation checks on a processed chunk."""
    checks = [
        validate_cqt_shape(cqt, CONFIG.n_bins),
        validate_piano_roll_shape(onset, 88),
        validate_piano_roll_shape(frame, 88),
        validate_data_ranges(onset, frame)
    ]
    
    if onset.shape[0] != frame.shape[0]:
        logger.error(f"Onset and frame time mismatch: {onset.shape[0]} vs {frame.shape[0]}")
        return False
    
    if cqt.shape[2] != onset.shape[0]:
        logger.error(f"CQT and roll time mismatch: {cqt.shape[2]} vs {onset.shape[0]}")
        return False
    
    return all(checks)

# ============================================================================
# CORE PREPROCESSING FUNCTIONS
# ============================================================================

def get_piano_cqt(file_path: str, config: PreprocessingConfig = CONFIG) -> np.ndarray:
    """
    Convert WAV file to normalized CQT spectrogram.
    
    Args:
        file_path: Path to audio file
        config: Preprocessing configuration
    
    Returns:
        CQT array of shape (1, n_bins, time_frames)
    """
    try:
        # Load audio
        signal, sr = librosa.load(file_path, sr=config.sample_rate)
        logger.debug(f"Loaded audio: {len(signal)} samples at {sr} Hz")
        
        # Compute CQT
        cqt_complex = librosa.cqt(
            signal,
            sr=sr,
            fmin=config.fmin,
            n_bins=config.n_bins,
            bins_per_octave=config.bins_per_octave,
            hop_length=config.hop_length
        )
        
        # Convert to magnitude and dB
        cqt_mag = np.abs(cqt_complex)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        
        # Standardize
        cqt_normalized = (cqt_db - config.global_mean) / config.global_std
        
        # Add channel dimension (1, n_bins, time)
        cqt_ready = cqt_normalized[np.newaxis, :, :]
        
        logger.debug(f"CQT shape: {cqt_ready.shape}")
        return cqt_ready
        
    except Exception as e:
        logger.error(f"Failed to process audio {file_path}: {e}")
        raise

def get_piano_roll_exact_alignment(
        midi_path: str,
        target_frames: int, # Exact number of CQT frames
        hop_length: int = 512,
        sr: int = 22050
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get piano roll with EXACT frame alignment to CQT.

    args:
        midi_path: Path to MIDI file
        target_frames: Exact number of frames (from CQT)
        hop_length: Must mach CQT hope_length
        sr: Must match CQT sample_rate
    Returns:
        (onset_roll, frame_roll) with exactly target_frames
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # Calculate time per frame (seconds)
    frame_time = hop_length / sr 

    # Create piano roll with EXACT target length
    times = np.arange(target_frames) * frame_time

    # Get piano roll at exact times
    piano_roll = np.zeros((128, target_frames), dtype=np.float32)

    for instruement in midi_data.instruments:
        if instruement.is_drum:
            continue

        for note in instruement.notes:
            # Find frames where this not is active
            start_frame = int(note.start / frame_time)
            end_frame = int(note.end / frame_time)

            # Clip to valid range
            start_frame = max(0, start_frame)
            end_frame = min(target_frames, end_frame)

            if start_frame < target_frames and end_frame > 0:
                # Mark not as active
                piano_roll[note.pitch, start_frame:end_frame] = note.velocity / 127.0

    # Extract piano range and transpose 
    piano_roll = piano_roll[21: 109, :].T

    # Create frame and onset rolls
    frame_roll = (piano_roll > 0).astype(np.float32)
    onset_roll = create_onset_roll(frame_roll, onset_length=3)

    return onset_roll, frame_roll

def get_piano_roll(
    midi_path: str, 
    config: PreprocessingConfig = CONFIG
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MIDI file to onset and frame rolls.
    
    Args:
        midi_path: Path to MIDI file
        config: Preprocessing configuration
    
    Returns:
        Tuple of (onset_roll, frame_roll), each shape (time_frames, 88)
    """
    try:
        # Load MIDI
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Calculate frame rate
        fps = config.sample_rate / config.hop_length
        
        # Get piano roll
        full_piano_roll = midi_data.get_piano_roll(fs=fps)
        
        # Extract piano range (MIDI 21-108 = 88 keys)
        piano_roll = full_piano_roll[21:109, :].T
        
        # Create binary frame roll
        frame_roll = (piano_roll > 0).astype(np.float32)
        
        # Create onset roll
        onset_roll = create_onset_roll(frame_roll, config.onset_length)
        
        logger.debug(f"Piano rolls shape: {onset_roll.shape}")
        return onset_roll, frame_roll
        
    except Exception as e:
        logger.error(f"Failed to process MIDI {midi_path}: {e}")
        raise

def create_onset_roll(frame_roll: np.ndarray, onset_length: int = 3) -> np.ndarray:
    """
    Create onset roll from frame roll by detecting note starts.
    
    Args:
        frame_roll: Binary matrix (time_frames, 88)
        onset_length: Number of frames to mark as onset
    
    Returns:
        Onset roll (time_frames, 88)
    """
    time_frames, num_keys = frame_roll.shape
    onset_roll = np.zeros_like(frame_roll)
    
    for key in range(num_keys):
        # Detect 0->1 transitions
        padded = np.pad(frame_roll[:, key], (1, 0), mode='constant')
        diff = np.diff(padded)
        onset_indices = np.where(diff == 1)[0]
        
        # Mark onset_length frames
        for onset_idx in onset_indices:
            end_idx = min(onset_idx + onset_length, time_frames)
            onset_roll[onset_idx:end_idx, key] = 1.0
    
    return onset_roll

def align_dimensions(
    cqt: np.ndarray, 
    onset_roll: np.ndarray, 
    frame_roll: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align time dimensions by trimming to minimum length.
    
    Args:
        cqt: Shape (1, n_bins, time)
        onset_roll: Shape (time, 88)
        frame_roll: Shape (time, 88)
    
    Returns:
        Aligned (cqt, onset_roll, frame_roll)
    """
    cqt_time = cqt.shape[2]
    roll_time = onset_roll.shape[0]
    
    if cqt_time != roll_time:
        min_time = min(cqt_time, roll_time)
        logger.warning(f"Aligning dimensions: CQT={cqt_time}, rolls={roll_time} -> {min_time}")
        
        cqt = cqt[:, :, :min_time]
        onset_roll = onset_roll[:min_time, :]
        frame_roll = frame_roll[:min_time, :]
    
    return cqt, onset_roll, frame_roll

def chunk_data(
    cqt: np.ndarray,
    onset_roll: np.ndarray,
    frame_roll: np.ndarray,
    chunk_length: int,
    hop_length: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split data into fixed-length chunks.
    
    Args:
        cqt: Shape (1, n_bins, time)
        onset_roll: Shape (time, 88)
        frame_roll: Shape (time, 88)
        chunk_length: Frames per chunk
        hop_length: Frames between chunk starts
    
    Returns:
        Lists of (cqt_chunks, onset_chunks, frame_chunks)
    """
    _, _, n_frames = cqt.shape
    
    cqt_chunks = []
    onset_chunks = []
    frame_chunks = []
    
    for start in range(0, n_frames - chunk_length + 1, hop_length):
        end = start + chunk_length
        
        cqt_chunks.append(cqt[:, :, start:end])
        onset_chunks.append(onset_roll[start:end, :])
        frame_chunks.append(frame_roll[start:end, :])
    
    return cqt_chunks, onset_chunks, frame_chunks

def process_file(
    audio_path: str,
    midi_path: str,
    config: PreprocessingConfig = CONFIG
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Process single audio-MIDI pair into training chunks.
    
    Args:
        audio_path: Path to WAV file
        midi_path: Path to MIDI file
        config: Preprocessing configuration
    
    Returns:
        Tuple of (cqt_chunks, onset_chunks, frame_chunks)
    """
    try:
        # Get CQT and piano rolls
        cqt = get_piano_cqt(audio_path, config)
        onset_roll, frame_roll = get_piano_roll(midi_path, config)
        
        # Align dimensions
        cqt, onset_roll, frame_roll = align_dimensions(cqt, onset_roll, frame_roll)
        
        # Validate before chunking
        if not validate_processed_chunk(cqt, onset_roll, frame_roll):
            logger.error(f"Validation failed for {audio_path}")
            return [], [], []
        
        # Chunk the data
        chunks = chunk_data(
            cqt, onset_roll, frame_roll,
            config.chunk_length,
            config.hop_length_chunks
        )
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}", exc_info=True)
        return [], [], []

# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

class ProcessingMetadata:
    """Track preprocessing progress and metadata."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metadata_file = output_dir / 'metadata.json'
        self.processed_files = set()
        self.chunk_index = 0
        self.file_to_chunks = {}  # Maps filename to list of chunk indices
        
        self.load()
    
    def load(self):
        """Load existing metadata if available."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.processed_files = set(data.get('processed_files', []))
                self.chunk_index = data.get('chunk_index', 0)
                self.file_to_chunks = data.get('file_to_chunks', {})
            logger.info(f"Loaded metadata: {len(self.processed_files)} files, {self.chunk_index} chunks")
    
    def save(self):
        """Save metadata to disk."""
        data = {
            'processed_files': list(self.processed_files),
            'chunk_index': self.chunk_index,
            'file_to_chunks': self.file_to_chunks,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def mark_processed(self, filename: str, num_chunks: int):
        """Mark a file as processed."""
        self.processed_files.add(filename)
        chunk_start = self.chunk_index
        self.file_to_chunks[filename] = list(range(chunk_start, chunk_start + num_chunks))
        self.chunk_index += num_chunks
        self.save()
    
    def is_processed(self, filename: str) -> bool:
        """Check if file has been processed."""
        return filename in self.processed_files

# ============================================================================
# DATASET LOADING FUNCTIONS
# ============================================================================

def find_audio_midi_pairs(dataset_path: Path) -> List[Tuple[Path, Path]]:
    """
    Find all valid audio-MIDI pairs in dataset.
    
    Args:
        dataset_path: Root directory of MAESTRO dataset
    
    Returns:
        List of (audio_path, midi_path) tuples
    """
    pairs = []
    
    # Find all year subdirectories
    year_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    for year_dir in year_dirs:
        for wav_path in year_dir.glob('*.wav'):
            # Try both .midi and .mid extensions
            midi_path = wav_path.with_suffix('.midi')
            if not midi_path.exists():
                midi_path = wav_path.with_suffix('.mid')
            
            if midi_path.exists():
                pairs.append((wav_path, midi_path))
            else:
                logger.warning(f"No MIDI file for {wav_path.name}")
    
    return pairs

def load_dataset(
    dataset_path: str,
    max_files: Optional[int] = None,
    config: PreprocessingConfig = CONFIG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset into memory (for small experiments).
    
    Args:
        dataset_path: Path to MAESTRO dataset
        max_files: Maximum files to process (None = all)
        config: Preprocessing configuration
    
    Returns:
        Tuple of (X, y_onset, y_frame) as numpy arrays
    """
    logger.info("Loading dataset into memory...")
    
    dataset_path = Path(dataset_path)
    pairs = find_audio_midi_pairs(dataset_path)
    
    if max_files:
        pairs = pairs[:max_files]
    
    logger.info(f"Processing {len(pairs)} audio-MIDI pairs")
    
    X_all = []
    y_onset_all = []
    y_frame_all = []
    
    # Process with progress bar
    for wav_path, midi_path in tqdm(pairs, desc="Processing files"):
        cqt_chunks, onset_chunks, frame_chunks = process_file(
            str(wav_path), str(midi_path), config
        )
        
        if len(cqt_chunks) > 0:
            X_all.extend(cqt_chunks)
            y_onset_all.extend(onset_chunks)
            y_frame_all.extend(frame_chunks)
    
    # Convert to arrays
    X = np.array(X_all, dtype=np.float32)
    y_onset = np.array(y_onset_all, dtype=np.float32)
    y_frame = np.array(y_frame_all, dtype=np.float32)
    
    logger.info(f"Dataset loaded: {len(X)} samples")
    logger.info(f"  CQT shape: {X.shape}")
    logger.info(f"  Onset shape: {y_onset.shape}")
    logger.info(f"  Frame shape: {y_frame.shape}")
    
    return X, y_onset, y_frame

def load_dataset_incremental(
    dataset_path: str,
    output_dir: str = 'processed_chunks',
    resume: bool = False,
    max_files: Optional[int] = None,
    config: PreprocessingConfig = CONFIG
) -> int:
    """
    Process dataset incrementally, saving chunks to disk.
    
    Args:
        dataset_path: Path to MAESTRO dataset
        output_dir: Directory to save processed chunks
        resume: Resume from previous run if True
        max_files: Maximum files to process (None = all)
        config: Preprocessing configuration
    
    Returns:
        Total number of chunks processed
    """
    logger.info("Starting incremental dataset processing...")
    
    # Setup paths
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    config.save(str(output_dir / 'config.json'))
    logger.info(f"Saved config to {output_dir / 'config.json'}")
    
    # Load metadata
    metadata = ProcessingMetadata(output_dir)
    
    if resume:
        logger.info(f"Resuming from {len(metadata.processed_files)} processed files")
    else:
        logger.info("Starting fresh preprocessing")
    
    # Find all pairs
    pairs = find_audio_midi_pairs(dataset_path)
    
    if max_files:
        pairs = pairs[:max_files]
    
    logger.info(f"Found {len(pairs)} audio-MIDI pairs")
    
    # Filter already processed (if resuming)
    if resume:
        pairs = [(w, m) for w, m in pairs if w.name not in metadata.processed_files]
        logger.info(f"Skipping {len(metadata.processed_files)} already processed files")
        logger.info(f"Processing {len(pairs)} remaining files")
    
    # Process files with progress bar
    for wav_path, midi_path in tqdm(pairs, desc="Processing files"):
        # Process file
        cqt_chunks, onset_chunks, frame_chunks = process_file(
            str(wav_path), str(midi_path), config
        )
        
        if len(cqt_chunks) == 0:
            logger.warning(f"No chunks generated for {wav_path.name}")
            continue
        
        # Save each chunk
        for i in range(len(cqt_chunks)):
            chunk_file = output_dir / f'chunk_{metadata.chunk_index:06d}.npz'
            
            np.savez_compressed(
                chunk_file,
                cqt=cqt_chunks[i],
                onset=onset_chunks[i],
                frame=frame_chunks[i],
                source_file=wav_path.name
            )
            
            metadata.chunk_index += 1
        
        # Update metadata
        metadata.mark_processed(wav_path.name, len(cqt_chunks))
        
        logger.debug(f"Processed {wav_path.name}: {len(cqt_chunks)} chunks")
    
    logger.info(f"✅ Processing complete: {metadata.chunk_index} total chunks")
    logger.info(f"Saved to {output_dir}")
    
    return metadata.chunk_index

# ============================================================================
# STATISTICS AND ANALYSIS
# ============================================================================

def compute_dataset_statistics(output_dir: str):
    """
    Compute and display statistics about processed dataset.
    
    Args:
        output_dir: Directory containing processed chunks
    """
    logger.info("Computing dataset statistics...")
    
    output_dir = Path(output_dir)
    chunk_files = sorted(output_dir.glob('chunk_*.npz'))
    
    if len(chunk_files) == 0:
        logger.warning("No chunks found")
        return
    
    # Sample statistics from first 100 chunks
    sample_size = min(100, len(chunk_files))
    
    onset_sparsity = []
    frame_sparsity = []
    polyphony = []
    
    for chunk_file in tqdm(chunk_files[:sample_size], desc="Computing stats"):
        data = np.load(chunk_file)
        
        onset_sparsity.append((data['onset'] > 0).mean())
        frame_sparsity.append((data['frame'] > 0).mean())
        polyphony.append(data['frame'].sum(axis=1).mean())
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    print(f"Total chunks: {len(chunk_files)}")
    print(f"Total size: {sum(f.stat().st_size for f in chunk_files) / 1e9:.2f} GB")
    print(f"\nSparsity (% of active values):")
    print(f"  Onset sparsity: {np.mean(onset_sparsity):.2%}")
    print(f"  Frame sparsity: {np.mean(frame_sparsity):.2%}")
    print(f"\nPolyphony:")
    print(f"  Average simultaneous notes: {np.mean(polyphony):.2f}")
    print(f"  Max simultaneous notes: {max(polyphony):.2f}")
    print("="*60 + "\n")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess MAESTRO dataset for piano transcription'
    )
    
    parser.add_argument(
        'dataset_path',
        help='Path to maestro-v3.0.0 directory'
    )
    
    parser.add_argument(
        '--mode',
        choices=['test', 'incremental'],
        default='test',
        help='Processing mode: test (load to RAM) or incremental (save to disk)'
    )
    
    parser.add_argument(
        '--output',
        default='processed_chunks',
        help='Output directory for incremental mode (default: processed_chunks)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (default: all)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted preprocessing (incremental mode only)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Compute dataset statistics after processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output) if args.mode == 'incremental' else None
    setup_logging(output_dir, args.verbose)
    
    logger.info("="*60)
    logger.info("Notes to Keys - Audio Preprocessing")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Dataset: {args.dataset_path}")
    
    # Process dataset
    if args.mode == 'test':
        X, y_onset, y_frame = load_dataset(
            args.dataset_path,
            max_files=args.max_files
        )
        
        # Optionally save
        if args.output:
            output_path = Path(args.output) / 'test_data.npz'
            output_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(output_path, X=X, y_onset=y_onset, y_frame=y_frame)
            logger.info(f"Saved test data to {output_path}")
    
    elif args.mode == 'incremental':
        total_chunks = load_dataset_incremental(
            args.dataset_path,
            output_dir=args.output,
            resume=args.resume,
            max_files=args.max_files
        )
        
        logger.info(f"Processed {total_chunks} chunks")
        
        # Compute statistics if requested
        if args.stats:
            compute_dataset_statistics(args.output)
    
    logger.info("✅ Processing complete!")

if __name__ == "__main__":
    main()