"""
validate_data.py - Validate preprocessed piano transcription data

This script checks if your preprocessed data is ready for training.
It verifies data quality, alignment, and creates visualizations.

Usage:
    python validate_data.py processed_chunks/test_data.npz

What this script does:
1. Loads your preprocessed data
2. Checks for common errors (NaN, wrong shapes, etc.)
3. Calculates statistics to verify quality
4. Creates visualizations to inspect alignment
5. Gives you a GO/NO-GO decision for training
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ============================================================================
# STEP 1: Load the Data
# ============================================================================

def load_data(npz_path):
    """
    Load preprocessed data from .npz file.
    
    WHY: Before training, we need to verify the data actually loaded correctly
    and has the expected structure.
    
    Args:
        npz_path: Path to the .npz file created by preprocessing
    
    Returns:
        X, y_onset, y_frame as numpy arrays
    """
    print("=" * 60)
    print("STEP 1: Loading Preprocessed Data")
    print("=" * 60)
    
    if not Path(npz_path).exists():
        print(f"❌ ERROR: File not found: {npz_path}")
        sys.exit(1)
    
    try:
        data = np.load(npz_path)
        X = data['X']           # CQT spectrograms
        y_onset = data['y_onset']  # Onset targets
        y_frame = data['y_frame']  # Frame targets
        
        print(f"✅ Successfully loaded data from {npz_path}")
        print(f"\nData shapes:")
        print(f"  CQT (X):        {X.shape}")
        print(f"  Onset targets:  {y_onset.shape}")
        print(f"  Frame targets:  {y_frame.shape}")
        
        # EXPLANATION:
        # X shape: (samples, channels, frequency_bins, time_frames)
        #   - samples: Number of training chunks (should be ~71,000+)
        #   - channels: Always 1 (mono audio)
        #   - frequency_bins: 176 (88 piano keys × 2 bins per semitone)
        #   - time_frames: 100 (length of each chunk)
        #
        # y_onset/y_frame shape: (samples, time_frames, piano_keys)
        #   - samples: Same number as X
        #   - time_frames: 100 (must match X!)
        #   - piano_keys: 88 (standard piano range)
        
        return X, y_onset, y_frame
        
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        sys.exit(1)

# ============================================================================
# STEP 2: Check Data Shapes
# ============================================================================

def validate_shapes(X, y_onset, y_frame):
    """
    Verify that all arrays have correct dimensions.
    
    WHY: If shapes are wrong, your model will crash during training.
    Better to catch this now than waste hours training only to crash.
    
    Args:
        X, y_onset, y_frame: Loaded data arrays
    
    Returns:
        True if all shapes are correct, False otherwise
    """
    print("\n" + "=" * 60)
    print("STEP 2: Validating Shapes")
    print("=" * 60)
    
    issues = []
    
    # Expected shapes
    n_samples = X.shape[0]
    expected_X_shape = (n_samples, 1, 176, 100)
    expected_onset_shape = (n_samples, 100, 88)
    expected_frame_shape = (n_samples, 100, 88)
    
    # Check X (CQT)
    if X.shape != expected_X_shape:
        issues.append(f"X shape is {X.shape}, expected {expected_X_shape}")
    
    # Check y_onset
    if y_onset.shape != expected_onset_shape:
        issues.append(f"Onset shape is {y_onset.shape}, expected {expected_onset_shape}")
    
    # Check y_frame
    if y_frame.shape != expected_frame_shape:
        issues.append(f"Frame shape is {y_frame.shape}, expected {expected_frame_shape}")
    
    # Check sample count matches
    if not (X.shape[0] == y_onset.shape[0] == y_frame.shape[0]):
        issues.append(f"Sample counts don't match: X={X.shape[0]}, onset={y_onset.shape[0]}, frame={y_frame.shape[0]}")
    
    # Check time dimension matches
    if X.shape[3] != y_onset.shape[1]:
        issues.append(f"Time dimensions don't match: X has {X.shape[3]} frames, onset has {y_onset.shape[1]}")
    
    if X.shape[3] != y_frame.shape[1]:
        issues.append(f"Time dimensions don't match: X has {X.shape[3]} frames, frame has {y_frame.shape[1]}")
    
    # Report results
    if issues:
        print("❌ SHAPE VALIDATION FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All shapes are correct!")
        print("\nEXPLANATION:")
        print("  - CQT has 1 channel, 176 frequency bins, 100 time frames per sample")
        print("  - Onset and Frame targets have 88 keys, 100 time frames per sample")
        print("  - Time dimensions match between CQT and targets (critical for alignment!)")
        return True

# ============================================================================
# STEP 3: Check for Bad Values (NaN, Inf)
# ============================================================================

def check_for_bad_values(X, y_onset, y_frame):
    """
    Check for NaN (Not a Number) and Inf (Infinity) values.
    
    WHY: NaN and Inf values will break training. They often indicate:
    - Division by zero in preprocessing
    - Logarithm of zero or negative number
    - Numerical overflow
    
    Args:
        X, y_onset, y_frame: Data arrays
    
    Returns:
        True if no bad values found, False otherwise
    """
    print("\n" + "=" * 60)
    print("STEP 3: Checking for Bad Values (NaN, Inf)")
    print("=" * 60)
    
    issues = []
    
    # Check X
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        issues.append(f"Found {nan_count:,} NaN values in CQT")
    
    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        issues.append(f"Found {inf_count:,} Inf values in CQT")
    
    # Check y_onset
    if np.isnan(y_onset).any():
        nan_count = np.isnan(y_onset).sum()
        issues.append(f"Found {nan_count:,} NaN values in onset targets")
    
    # Check y_frame
    if np.isnan(y_frame).any():
        nan_count = np.isnan(y_frame).sum()
        issues.append(f"Found {nan_count:,} NaN values in frame targets")
    
    # Report results
    if issues:
        print("❌ FOUND BAD VALUES:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nWHAT TO DO:")
        print("  - Re-run preprocessing with debugging enabled")
        print("  - Check for division by zero or log of negative numbers")
        return False
    else:
        print("✅ No NaN or Inf values found!")
        print("\nEXPLANATION:")
        print("  - All values are valid numbers (no NaN/Inf)")
        print("  - Data is safe for neural network training")
        return True

# ============================================================================
# STEP 4: Check Data Ranges
# ============================================================================

def validate_data_ranges(X, y_onset, y_frame):
    """
    Verify that data values are in expected ranges.
    
    WHY: Neural networks work best when:
    - Input features (CQT) are normalized (centered around 0, std ~1)
    - Target labels (onset/frame) are binary (0 or 1)
    
    Args:
        X, y_onset, y_frame: Data arrays
    
    Returns:
        True if ranges are reasonable, False if problematic
    """
    print("\n" + "=" * 60)
    print("STEP 4: Validating Data Ranges")
    print("=" * 60)
    
    # CQT statistics
    print("\nCQT (Input Features):")
    print(f"  Min:  {X.min():.4f}")
    print(f"  Max:  {X.max():.4f}")
    print(f"  Mean: {X.mean():.4f}")
    print(f"  Std:  {X.std():.4f}")
    
    # EXPLANATION:
    # If using z-score normalization (subtract mean, divide by std):
    #   - Mean should be close to 0
    #   - Std should be close to 1
    #   - Min/Max should be roughly -3 to +3
    #
    # If using min-max normalization:
    #   - Min should be 0
    #   - Max should be 1
    
    # Check if CQT is normalized
    cqt_normalized = False
    if abs(X.mean()) < 0.5 and abs(X.std() - 1.0) < 0.5:
        print("  ✅ Looks like z-score normalized (good!)")
        cqt_normalized = True
    elif X.min() >= 0 and X.max() <= 1.5:
        print("  ✅ Looks like min-max normalized (good!)")
        cqt_normalized = True
    else:
        print("  ⚠️  Warning: CQT doesn't appear normalized")
        print("     This might slow down training")
    
    # Onset/Frame statistics
    print("\nOnset Targets:")
    print(f"  Min:       {y_onset.min():.4f}")
    print(f"  Max:       {y_onset.max():.4f}")
    print(f"  Unique:    {np.unique(y_onset)}")
    print(f"  Sparsity:  {(y_onset > 0).mean():.2%}")
    
    print("\nFrame Targets:")
    print(f"  Min:       {y_frame.min():.4f}")
    print(f"  Max:       {y_frame.max():.4f}")
    print(f"  Unique:    {np.unique(y_frame)}")
    print(f"  Sparsity:  {(y_frame > 0).mean():.2%}")
    
    # EXPLANATION:
    # Onset/Frame should be binary (only 0 and 1)
    # Sparsity = percentage of active values
    #   - Onset: 1-3% (most frames don't have note starts)
    #   - Frame: 10-20% (notes held over multiple frames)
    
    # Check if targets are binary
    onset_binary = set(np.unique(y_onset)).issubset({0.0, 1.0})
    frame_binary = set(np.unique(y_frame)).issubset({0.0, 1.0})
    
    issues = []
    
    if not onset_binary:
        issues.append("Onset targets are not binary (should only be 0 or 1)")
    
    if not frame_binary:
        issues.append("Frame targets are not binary (should only be 0 or 1)")
    
    # Check sparsity
    onset_sparsity = (y_onset > 0).mean()
    frame_sparsity = (y_frame > 0).mean()
    
    if onset_sparsity < 0.001 or onset_sparsity > 0.1:
        issues.append(f"Onset sparsity {onset_sparsity:.2%} seems unusual (expected 1-3%)")
    
    if frame_sparsity < 0.05 or frame_sparsity > 0.3:
        issues.append(f"Frame sparsity {frame_sparsity:.2%} seems unusual (expected 10-20%)")
    
    if issues:
        print("\n⚠️  WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ Data ranges look good!")
        return True

# ============================================================================
# STEP 5: Calculate Dataset Statistics
# ============================================================================

def compute_statistics(X, y_onset, y_frame):
    """
    Compute useful statistics about the dataset.
    
    WHY: Understanding your data helps you:
    - Set appropriate loss function weights (for imbalanced classes)
    - Know what to expect during training
    - Debug if results are unexpected
    
    Args:
        X, y_onset, y_frame: Data arrays
    """
    print("\n" + "=" * 60)
    print("STEP 5: Dataset Statistics")
    print("=" * 60)
    
    n_samples = X.shape[0]
    n_frames_per_sample = X.shape[3]
    fps = 43.1  # Frames per second
    
    # Total duration
    total_frames = n_samples * n_frames_per_sample
    total_seconds = total_frames / fps
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    
    print(f"\nDataset Size:")
    print(f"  Total samples:  {n_samples:,}")
    print(f"  Total frames:   {total_frames:,}")
    print(f"  Total duration: {total_hours:.2f} hours ({total_minutes:.1f} minutes)")
    
    # Polyphony (average simultaneous notes)
    avg_polyphony = y_frame.sum(axis=2).mean()
    max_polyphony = y_frame.sum(axis=2).max()
    
    print(f"\nPolyphony (simultaneous notes):")
    print(f"  Average: {avg_polyphony:.2f} notes")
    print(f"  Maximum: {int(max_polyphony)} notes")
    
    # EXPLANATION:
    # Polyphony = how many notes are played at once
    # Piano music typically has 2-5 simultaneous notes
    # Complex pieces might have 8-10 notes
    
    # Most/least common notes
    note_activity = y_frame.sum(axis=(0, 1))  # Sum across samples and time
    most_active_note = note_activity.argmax()
    least_active_note = note_activity.argmin()
    
    print(f"\nNote Activity:")
    print(f"  Most common note:  #{most_active_note} (MIDI {most_active_note + 21})")
    print(f"  Least common note: #{least_active_note} (MIDI {least_active_note + 21})")
    
    # EXPLANATION:
    # MIDI notes 21-108 = piano range (A0 to C8)
    # Middle C = MIDI 60 = note #39 in our 88-key array
    # Most music centers around middle octaves
    
    # File size estimate
    bytes_per_element = 4  # float32
    total_bytes = (X.size + y_onset.size + y_frame.size) * bytes_per_element
    total_gb = total_bytes / (1024**3)
    
    print(f"\nStorage:")
    print(f"  Uncompressed size: {total_gb:.2f} GB")
    print(f"  Compressed (.npz): ~{total_gb * 0.6:.2f} GB (estimate)")

# ============================================================================
# STEP 6: Visualize Sample Data
# ============================================================================

def visualize_samples(X, y_onset, y_frame, output_dir='validation_plots'):
    """
    Create visualizations to inspect data quality and alignment.
    
    WHY: Looking at your data helps you:
    - Verify that CQT and labels are actually aligned
    - Spot preprocessing errors visually
    - Understand what the model will "see"
    
    Args:
        X, y_onset, y_frame: Data arrays
        output_dir: Where to save plots
    """
    print("\n" + "=" * 60)
    print("STEP 6: Creating Visualizations")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find a sample with notes (not silent)
    # Look for samples where at least 5% of frames have active notes
    active_samples = (y_frame.sum(axis=(1, 2)) / (100 * 88)) > 0.05
    active_indices = np.where(active_samples)[0]
    
    if len(active_indices) == 0:
        print("⚠️  Warning: No active samples found (all silent?)")
        sample_idx = 0
    else:
        # Pick a random active sample
        sample_idx = np.random.choice(active_indices)
    
    print(f"\nVisualizing sample #{sample_idx}...")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: CQT Spectrogram
    im1 = axes[0].imshow(
        X[sample_idx, 0, :, :],  # Remove channel dimension
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    axes[0].set_title('CQT Spectrogram (Input to Model)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency Bin (176 bins)', fontsize=11)
    axes[0].set_xlabel('Time Frame (100 frames ≈ 2.3 seconds)', fontsize=11)
    plt.colorbar(im1, ax=axes[0], label='Normalized Amplitude')
    
    # Add explanation
    axes[0].text(
        0.02, 0.98, 
        'WHAT YOU SEE: Bright vertical lines = note onsets\nHorizontal bands = sustained notes',
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Plot 2: Onset Targets
    im2 = axes[1].imshow(
        y_onset[sample_idx].T,  # Transpose to (keys, time)
        aspect='auto',
        origin='lower',
        cmap='Reds',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )
    axes[1].set_title('Onset Targets (When Notes START)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Piano Key (88 keys)', fontsize=11)
    axes[1].set_xlabel('Time Frame', fontsize=11)
    plt.colorbar(im2, ax=axes[1], label='Onset (0 or 1)')
    
    # Add explanation
    axes[1].text(
        0.02, 0.98,
        'WHAT YOU SEE: Red vertical lines = when notes are first pressed\nThese should align with bright lines in CQT above',
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Plot 3: Frame Targets
    im3 = axes[2].imshow(
        y_frame[sample_idx].T,  # Transpose to (keys, time)
        aspect='auto',
        origin='lower',
        cmap='Blues',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )
    axes[2].set_title('Frame Targets (Note DURATIONS)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Piano Key (88 keys)', fontsize=11)
    axes[2].set_xlabel('Time Frame', fontsize=11)
    plt.colorbar(im3, ax=axes[2], label='Active (0 or 1)')
    
    # Add explanation
    axes[2].text(
        0.02, 0.98,
        'WHAT YOU SEE: Blue horizontal blocks = how long notes are held\nWider blocks = longer notes',
        transform=axes[2].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'sample_{sample_idx}_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_file}")
    
    # Create a second figure showing multiple samples
    print("\nCreating overview of multiple samples...")
    
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
    fig2.suptitle('Random Sample Overview (9 samples)', fontsize=16, fontweight='bold')
    
    # Pick 9 random active samples
    if len(active_indices) >= 9:
        random_indices = np.random.choice(active_indices, 9, replace=False)
    else:
        random_indices = np.random.choice(len(X), 9, replace=False)
    
    for i, idx in enumerate(random_indices):
        row = i // 3
        col = i % 3
        
        # Plot just the CQT for each sample
        axes2[row, col].imshow(
            X[idx, 0, :, :],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        axes2[row, col].set_title(f'Sample #{idx}', fontsize=10)
        axes2[row, col].set_xlabel('Time', fontsize=8)
        axes2[row, col].set_ylabel('Freq', fontsize=8)
        axes2[row, col].tick_params(labelsize=7)
    
    plt.tight_layout()
    
    output_file2 = output_dir / 'samples_overview.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved overview to: {output_file2}")
    
    print(f"\n📊 All visualizations saved to: {output_dir}/")

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def main():
    """
    Main validation function - runs all checks.
    """
    print("\n" + "=" * 60)
    print("NOTES TO KEYS - DATA VALIDATION")
    print("=" * 60)
    print("\nThis script validates your preprocessed data before training.")
    print("It will check for errors and create visualizations.\n")
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python validate_data.py <path_to_npz_file>")
        print("\nExample:")
        print("  python validate_data.py processed_chunks/test_data.npz")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    
    # Run all validation steps
    X, y_onset, y_frame = load_data(npz_path)
    
    shapes_ok = validate_shapes(X, y_onset, y_frame)
    no_bad_values = check_for_bad_values(X, y_onset, y_frame)
    ranges_ok = validate_data_ranges(X, y_onset, y_frame)
    
    compute_statistics(X, y_onset, y_frame)
    visualize_samples(X, y_onset, y_frame)
    
    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    all_checks_passed = shapes_ok and no_bad_values and ranges_ok
    
    if all_checks_passed:
        print("\n✅ ✅ ✅ DATA VALIDATION PASSED! ✅ ✅ ✅")
        print("\nYour data is ready for training!")
        print("\nNext steps:")
        print("  1. Review the visualizations in 'validation_plots/'")
        print("  2. Verify that CQT and onset labels are aligned")
        print("  3. Create PyTorch DataLoader")
        print("  4. Build and train your model!")
    else:
        print("\n❌ ❌ ❌ DATA VALIDATION FAILED ❌ ❌ ❌")
        print("\nSome checks failed. Review the errors above.")
        print("\nWhat to do:")
        print("  1. Check the warnings and errors")
        print("  2. Re-run preprocessing if needed")
        print("  3. Run this validation script again")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()