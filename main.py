import numpy as np
import librosa
import pretty_midi
import os
import sys
from pathlib import Path

# Global normalization constants for CQT (estimated from dataset)
GLOBAL_MEAN = -40.0  # dB
GLOBAL_STD = 20.0    # dB


def main():
    # Check for correct usage
    if len(sys.argv) not in [2, 3]:
        print("Usage: python main.py <path to maestro dataset> [max_files]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) == 3 else None
    
    # Load preprocessed dataset
    print("🎹 Loading MAESTRO dataset...\n")
    X_train, y_onset, y_frame = load_dataset(
        dataset_path, 
        chunk_length=100,
        hop_length=50,
        max_files=max_files
    )
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"  Total samples: {len(X_train)}")
    print(f"  CQT shape: {X_train.shape}")        # (n_samples, 1, 176, 100)
    print(f"  Onset shape: {y_onset.shape}")      # (n_samples, 100, 88)
    print(f"  Frame shape: {y_frame.shape}")      # (n_samples, 100, 88)
    
    # Save processed data (optional)
    print("\n💾 Saving processed data...")
    np.savez_compressed(
        'maestro_processed.npz',
        X=X_train,
        y_onset=y_onset,
        y_frame=y_frame
    )
    print("  Saved to maestro_processed.npz")



def get_piano_cqt(file_path, sr=22050, hop_length=512):
    """
    Converts a WAV file to a normalized CQT spectrogram for NN input.
    Uses 2 bins per semitone for better frequency resolution.
    
    Returns:
        cqt_normalized: numpy array of shape (1, 176, time_frames)
                       176 = 88 keys × 2 bins per semitone
    """
    # 1. Load audio
    signal, sample_rate = librosa.load(file_path, sr=sr)
    
    # 2. Compute CQT with 2 bins per semitone
    cqt_complex = librosa.cqt(
        signal,
        sr=sample_rate,
        fmin=librosa.note_to_hz('A0'),  # 27.5 Hz
        n_bins=88 * 2,                  # 176 bins (2 per semitone)
        bins_per_octave=24,             # 24 bins per octave (2 per semitone)
        hop_length=hop_length
    )
    
    # 3. Convert to magnitude
    cqt_mag = np.abs(cqt_complex)
    
    # 4. Convert to dB
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    
    # 5. Standardize using global statistics
    cqt_normalized = (cqt_db - GLOBAL_MEAN) / GLOBAL_STD
    
    # 6. Add channel dimension for PyTorch (1, 176, time)
    cqt_ready = cqt_normalized[np.newaxis, :, :]
    
    return cqt_ready


def get_piano_roll(midi_path, sr=22050, hop_length=512, onset_length=3):
    """
    Converts MIDI file to time-aligned onset and frame matrices.
    
    Args:
        midi_path: Path to MIDI file
        sr: Sample rate (must match audio)
        hop_length: Hop length (must match CQT)
        onset_length: Number of frames to mark as onset (default 3)
    
    Returns:
        onset_roll: numpy array of shape (time_frames, 88)
                   1 for first 'onset_length' frames of note, 0 otherwise
        frame_roll: numpy array of shape (time_frames, 88)
                   1 while note is active, 0 otherwise
    """
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Calculate frame rate (must match CQT)
    fps = sr / hop_length  # ~43.1 frames per second
    
    # Get piano roll (velocity values, 0-127)
    # Shape: (128, time_frames) - all MIDI notes
    full_piano_roll = midi_data.get_piano_roll(fs=fps)
    
    # Extract only piano range (MIDI notes 21-108 = 88 keys)
    # MIDI note 21 = A0, MIDI note 108 = C8
    piano_roll = full_piano_roll[21:109, :]  # Shape: (88, time_frames)
    
    # Transpose to (time_frames, 88) for easier processing
    piano_roll = piano_roll.T
    
    # Create binary frame roll (1 if note active, 0 otherwise)
    frame_roll = (piano_roll > 0).astype(np.float32)
    
    # Create onset roll
    onset_roll = create_onset_roll(frame_roll, onset_length)
    
    return onset_roll, frame_roll


def create_onset_roll(frame_roll, onset_length=3):
    """
    Creates onset roll from frame roll by detecting note starts.
    
    Args:
        frame_roll: (time_frames, 88) binary matrix
        onset_length: number of frames to mark as onset
    
    Returns:
        onset_roll: (time_frames, 88) binary matrix
    """
    time_frames, num_keys = frame_roll.shape
    onset_roll = np.zeros_like(frame_roll)
    
    for key in range(num_keys):
        # Find where notes start (0 -> 1 transition)
        # Pad with 0 at start to catch first frame onsets
        padded = np.pad(frame_roll[:, key], (1, 0), mode='constant')
        diff = np.diff(padded)
        
        # onset_indices: where diff == 1 (note started)
        onset_indices = np.where(diff == 1)[0]
        
        # Mark onset_length frames starting from each onset
        for onset_idx in onset_indices:
            end_idx = min(onset_idx + onset_length, time_frames)
            onset_roll[onset_idx:end_idx, key] = 1.0
    
    return onset_roll


def verify_alignment(cqt, onset_roll, frame_roll):
    """
    Verify that CQT and piano rolls have matching time dimensions.
    
    Args:
        cqt: (1, 176, time_frames)
        onset_roll: (time_frames, 88)
        frame_roll: (time_frames, 88)
    
    Returns:
        Boolean indicating if dimensions match
    """
    cqt_time = cqt.shape[2]
    onset_time = onset_roll.shape[0]
    frame_time = frame_roll.shape[0]
    
    if cqt_time == onset_time == frame_time:
        return True
    else:
        print(f"⚠️ Dimension mismatch!")
        print(f"  CQT time: {cqt_time}")
        print(f"  Onset time: {onset_time}")
        print(f"  Frame time: {frame_time}")
        return False


def align_dimensions(cqt, onset_roll, frame_roll):
    """
    Ensure all matrices have the same time dimension by trimming to minimum.
    
    Returns:
        Aligned cqt, onset_roll, frame_roll
    """
    cqt_time = cqt.shape[2]
    roll_time = onset_roll.shape[0]
    
    min_time = min(cqt_time, roll_time)
    
    # Trim to minimum length
    cqt_aligned = cqt[:, :, :min_time]
    onset_aligned = onset_roll[:min_time, :]
    frame_aligned = frame_roll[:min_time, :]
    
    return cqt_aligned, onset_aligned, frame_aligned


def chunk_data(cqt, onset_roll, frame_roll, chunk_length=100, hop_length=50):
    """
    Split aligned data into fixed-length chunks for training.
    
    Args:
        cqt: (1, 176, time_frames)
        onset_roll: (time_frames, 88)
        frame_roll: (time_frames, 88)
        chunk_length: number of frames per chunk
        hop_length: frames between chunk starts
    
    Returns:
        Lists of chunks for each input/target
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


def process_file(audio_path, midi_path, chunk_length=100, hop_length=50):
    """
    Process a single audio-MIDI pair into training samples.
    
    Returns:
        Tuple of (cqt_chunks, onset_chunks, frame_chunks)
    """
    try:
        # Get CQT
        cqt = get_piano_cqt(audio_path)
        
        # Get piano rolls
        onset_roll, frame_roll = get_piano_roll(midi_path)
        
        # Verify alignment
        if not verify_alignment(cqt, onset_roll, frame_roll):
            print(f"⚠️ Aligning dimensions for {os.path.basename(audio_path)}")
            cqt, onset_roll, frame_roll = align_dimensions(cqt, onset_roll, frame_roll)
        
        # Chunk the data
        cqt_chunks, onset_chunks, frame_chunks = chunk_data(
            cqt, onset_roll, frame_roll, 
            chunk_length=chunk_length, 
            hop_length=hop_length
        )
        
        return cqt_chunks, onset_chunks, frame_chunks
        
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return [], [], []


def load_dataset(dataset_path, chunk_length=100, hop_length=50, max_files=None):
    """
    Load and preprocess the MAESTRO dataset with year subdirectories.
    
    Args:
        dataset_path: Path to maestro-v3.0.0 folder
        chunk_length: Length of each training chunk (frames)
        hop_length: Frames between chunk starts
        max_files: Maximum number of files to process (None = all)
    
    Returns:
        X: CQT chunks, shape (n_samples, 1, 176, chunk_length)
        y_onset: Onset targets, shape (n_samples, chunk_length, 88)
        y_frame: Frame targets, shape (n_samples, chunk_length, 88)
    """
    X_all = []
    y_onset_all = []
    y_frame_all = []
    
    dataset_path = Path(dataset_path)
    
    # Find all year subdirectories
    year_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(year_dirs)} year directories: {[d.name for d in year_dirs]}")
    
    file_count = 0
    
    for year_dir in year_dirs:
        print(f"\n📁 Processing {year_dir.name}/")
        
        # Find all WAV files in this year
        wav_files = sorted(year_dir.glob('*.wav'))
        
        for wav_path in wav_files:
            # Construct corresponding MIDI path
            midi_path = wav_path.with_suffix('.midi')
            
            if not midi_path.exists():
                # Try .mid extension
                midi_path = wav_path.with_suffix('.mid')
                
            if not midi_path.exists():
                print(f"⚠️ No MIDI file for {wav_path.name}, skipping")
                continue
            
            print(f"  Processing {wav_path.name}...", end=' ')
            
            # Process this file
            cqt_chunks, onset_chunks, frame_chunks = process_file(
                str(wav_path), 
                str(midi_path), 
                chunk_length=chunk_length,
                hop_length=hop_length
            )
            
            if len(cqt_chunks) > 0:
                X_all.extend(cqt_chunks)
                y_onset_all.extend(onset_chunks)
                y_frame_all.extend(frame_chunks)
                print(f"✓ ({len(cqt_chunks)} chunks)")
            else:
                print("✗ (failed)")
            
            file_count += 1
            
            # Stop if we've hit max_files limit
            if max_files and file_count >= max_files:
                print(f"\n⏹️ Reached max_files limit ({max_files})")
                break
        
        if max_files and file_count >= max_files:
            break
    
    # Convert to numpy arrays
    X = np.array(X_all, dtype=np.float32)
    y_onset = np.array(y_onset_all, dtype=np.float32)
    y_frame = np.array(y_frame_all, dtype=np.float32)
    
    return X, y_onset, y_frame


if __name__ == "__main__":
    main()