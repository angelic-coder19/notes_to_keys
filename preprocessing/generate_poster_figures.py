"""
generate_poster_figures.py
Notes to Keys — Research Poster Figure Generator

Generates three publication-quality individual figures for the poster:
    1. cqt_input.png       — CQT spectrogram (the model input)
    2. onset_roll.png      — Onset targets (when notes start)
    3. frame_roll.png      — Frame targets (note durations)

Usage:
    python generate_poster_figures.py <path_to_npz_file> [--sample <index>]

Examples:
    python generate_poster_figures.py processed_chunks/test_data.npz
    python generate_poster_figures.py processed_chunks/test_data.npz --sample 33214

Output:
    Saves three .png files to a folder called poster_figures/
    in the same directory you run the script from.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys
import argparse


# ============================================================================
# CONFIGURATION — edit these if you want to change the output style
# ============================================================================

# Output folder name
OUTPUT_DIR = 'poster_figures'

# Figure dimensions in inches — sized for a 48×36 inch poster
# Each panel will print at roughly 8×5 inches at 300 DPI
FIG_WIDTH  = 8.5
FIG_HEIGHT = 4.5

# Resolution — 300 DPI is the standard minimum for print
DPI = 300

# Font sizes — tuned so text is readable when printed on a poster
TITLE_FONTSIZE   = 14
AXISLABEL_SIZE   = 11
TICKLABEL_SIZE   = 9

# Color maps — chosen for clarity and print legibility
CQT_CMAP    = 'viridis'   # standard for spectrograms
ONSET_CMAP  = 'Reds'
FRAME_CMAP  = 'Blues'

# Minimum note activity — samples below this threshold are skipped
# (avoids picking a silent or near-silent chunk)
MIN_ACTIVITY = 0.05


# ============================================================================
# LOAD DATA
# ============================================================================

def load_npz(npz_path):
    """
    Load the preprocessed .npz file produced by main2.py / validate_data.py.
    Supports both test_data.npz (X / y_onset / y_frame keys)
    and individual chunk files (cqt / onset / frame keys).
    """
    path = Path(npz_path)
    if not path.exists():
        print(f"ERROR: File not found — {npz_path}")
        sys.exit(1)

    data = np.load(npz_path)

    # Test_data.npz format (full dataset file)
    if 'X' in data:
        X       = data['X']        # (N, 1, 176, 100)
        y_onset = data['y_onset']  # (N, 100, 88)
        y_frame = data['y_frame']  # (N, 100, 88)
        print(f"Loaded test_data.npz — {len(X):,} samples")

    # Individual chunk format
    elif 'cqt' in data:
        # Wrap single chunk in a batch dimension for consistent indexing
        X       = data['cqt'][np.newaxis]      # (1, 1, 176, 100)
        y_onset = data['onset'][np.newaxis]    # (1, 100, 88)
        y_frame = data['frame'][np.newaxis]    # (1, 100, 88)
        print("Loaded single chunk file — 1 sample")

    else:
        print(f"ERROR: Unrecognised .npz format. Keys found: {list(data.keys())}")
        sys.exit(1)

    return X, y_onset, y_frame


# ============================================================================
# SAMPLE SELECTION
# ============================================================================

def pick_sample(X, y_frame, requested_index=None):
    """
    Choose which sample index to visualise.

    If a specific index is requested, use it.
    Otherwise find a sample with good musical content
    (at least MIN_ACTIVITY fraction of frames have active notes).
    """
    if requested_index is not None:
        idx = int(requested_index)
        if idx >= len(X):
            print(f"WARNING: Requested index {idx} exceeds dataset size "
                  f"({len(X)}). Using index 0 instead.")
            idx = 0
        print(f"Using requested sample index: {idx}")
        return idx

    # Find musically active samples
    activity = y_frame.sum(axis=(1, 2)) / (y_frame.shape[1] * y_frame.shape[2])
    active   = np.where(activity > MIN_ACTIVITY)[0]

    if len(active) == 0:
        print("WARNING: No active samples found — using index 0.")
        return 0

    # Pick the sample closest to the median activity level
    # (avoids picking an unusually dense or unusually sparse chunk)
    median_activity = np.median(activity[active])
    idx = active[np.argmin(np.abs(activity[active] - median_activity))]
    print(f"Auto-selected sample index: {idx}  "
          f"(activity: {activity[idx]*100:.1f}%)")
    return idx


# ============================================================================
# FIGURE 1 — CQT INPUT
# ============================================================================

def plot_cqt(X, sample_idx, output_dir):
    """
    Plot the CQT spectrogram for a single sample.

    What this shows:
    - Y axis: 176 frequency bins (88 piano keys × 2 bins per semitone)
    - X axis: 100 time frames (~2.3 seconds of audio)
    - Colour: normalised amplitude (viridis colourmap)
    - Bright horizontal bands = notes being held
    - Brighter vertical regions = higher overall energy (chord onsets)
    """
    cqt = X[sample_idx, 0, :, :]   # shape: (176, 100) — remove channel dim

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    img = ax.imshow(
        cqt,
        aspect='auto',
        origin='lower',            # low frequency bins at the bottom
        cmap=CQT_CMAP,
        interpolation='nearest',
    )

    # Colourbar
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label('Normalised Amplitude', fontsize=AXISLABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICKLABEL_SIZE)

    # Axes labels
    ax.set_title('CQT Spectrogram — Model Input',
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    ax.set_xlabel('Time Frame  (100 frames ≈ 2.3 s)',
                  fontsize=AXISLABEL_SIZE)
    ax.set_ylabel('Frequency Bin  (176 bins = 88 keys × 2)',
                  fontsize=AXISLABEL_SIZE)
    ax.tick_params(labelsize=TICKLABEL_SIZE)

    # Add octave reference lines on the Y axis
    # Each piano octave spans 24 bins (2 bins/semitone × 12 semitones)
    # A0 = bin 0, A1 = bin 24, A2 = bin 48 ... A7 = bin 168
    octave_bins   = [0, 24, 48, 72, 96, 120, 144, 168]
    octave_labels = ['A0\n27.5 Hz', 'A1\n55 Hz', 'A2\n110 Hz',
                     'A3\n220 Hz', 'A4\n440 Hz', 'A5\n880 Hz',
                     'A6\n1760 Hz', 'A7\n3520 Hz']

    ax.set_yticks(octave_bins)
    ax.set_yticklabels(octave_labels, fontsize=8)

    # Thin horizontal dashed lines at octave boundaries
    for b in octave_bins:
        ax.axhline(y=b, color='white', linewidth=0.4,
                   linestyle='--', alpha=0.35)

    plt.tight_layout()

    out_path = output_dir / 'cqt_input.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# FIGURE 2 — ONSET ROLL
# ============================================================================

def plot_onset(y_onset, sample_idx, output_dir):
    """
    Plot the onset roll (ground-truth note start targets).

    What this shows:
    - Y axis: 88 piano keys (key 0 = A0, key 87 = C8)
    - X axis: 100 time frames
    - Each short red mark = the moment a key was pressed
    - The first 3 frames of each note are marked (onset_length=3 in config)
    - These should align vertically with bright events in the CQT
    """
    onset = y_onset[sample_idx].T   # shape: (88, 100) — transpose to (keys, time)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    img = ax.imshow(
        onset,
        aspect='auto',
        origin='lower',
        cmap=ONSET_CMAP,
        interpolation='nearest',
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label('Onset  (0 = inactive, 1 = onset)', fontsize=AXISLABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICKLABEL_SIZE)

    ax.set_title('Onset Roll — When Each Note Starts',
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    ax.set_xlabel('Time Frame  (100 frames ≈ 2.3 s)',
                  fontsize=AXISLABEL_SIZE)
    ax.set_ylabel('Piano Key  (0 = A0,  87 = C8)',
                  fontsize=AXISLABEL_SIZE)
    ax.tick_params(labelsize=TICKLABEL_SIZE)

    # Octave reference lines on Y axis
    # Piano key 0 = A0. Each octave = 12 keys.
    # A0=key0, A1=key12, A2=key24, A3=key36, A4=key48,
    # A5=key60, A6=key72, A7=key84
    octave_keys   = [0, 12, 24, 36, 48, 60, 72, 84]
    octave_labels = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

    ax.set_yticks(octave_keys)
    ax.set_yticklabels(octave_labels, fontsize=TICKLABEL_SIZE)

    for k in octave_keys:
        ax.axhline(y=k, color='gray', linewidth=0.4,
                   linestyle='--', alpha=0.4)

    plt.tight_layout()

    out_path = output_dir / 'onset_roll.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# FIGURE 3 — FRAME ROLL
# ============================================================================

def plot_frame(y_frame, sample_idx, output_dir):
    """
    Plot the frame roll (ground-truth note duration targets).

    What this shows:
    - Y axis: 88 piano keys
    - X axis: 100 time frames
    - Each blue horizontal block = a key being held down
    - Wider blocks = longer note durations
    - Corresponds directly to the MIDI note data from MAESTRO
    """
    frame = y_frame[sample_idx].T   # shape: (88, 100)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    img = ax.imshow(
        frame,
        aspect='auto',
        origin='lower',
        cmap=FRAME_CMAP,
        interpolation='nearest',
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label('Frame Active  (0 = released, 1 = held)',
                   fontsize=AXISLABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICKLABEL_SIZE)

    ax.set_title('Frame Roll — How Long Each Note Is Held',
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    ax.set_xlabel('Time Frame  (100 frames ≈ 2.3 s)',
                  fontsize=AXISLABEL_SIZE)
    ax.set_ylabel('Piano Key  (0 = A0,  87 = C8)',
                  fontsize=AXISLABEL_SIZE)
    ax.tick_params(labelsize=TICKLABEL_SIZE)

    # Same octave reference lines as onset plot for consistency
    octave_keys   = [0, 12, 24, 36, 48, 60, 72, 84]
    octave_labels = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

    ax.set_yticks(octave_keys)
    ax.set_yticklabels(octave_labels, fontsize=TICKLABEL_SIZE)

    for k in octave_keys:
        ax.axhline(y=k, color='gray', linewidth=0.4,
                   linestyle='--', alpha=0.4)

    plt.tight_layout()

    out_path = output_dir / 'frame_roll.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate poster figures for Notes to Keys research poster.'
    )
    parser.add_argument(
        'npz_path',
        help='Path to preprocessed .npz file '
             '(e.g. processed_chunks/test_data.npz)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Specific sample index to visualise (default: auto-selected)'
    )
    parser.add_argument(
        '--output',
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Notes to Keys — Poster Figure Generator")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {args.npz_path}")
    X, y_onset, y_frame = load_npz(args.npz_path)
    print(f"  CQT shape:   {X.shape}")
    print(f"  Onset shape: {y_onset.shape}")
    print(f"  Frame shape: {y_frame.shape}")

    # Select sample
    print()
    sample_idx = pick_sample(X, y_frame, args.sample)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to: {output_dir}/")
    print()

    # Generate the three figures
    plot_cqt(X, sample_idx, output_dir)
    plot_onset(y_onset, sample_idx, output_dir)
    plot_frame(y_frame, sample_idx, output_dir)

    print()
    print("=" * 60)
    print("  Done. Three figures saved:")
    print(f"    {output_dir}/cqt_input.png")
    print(f"    {output_dir}/onset_roll.png")
    print(f"    {output_dir}/frame_roll.png")
    print()
    print("  Poster placement guide:")
    print("    cqt_input.png  — Results section, top panel")
    print("    onset_roll.png — Results section, middle panel")
    print("    frame_roll.png — Results section, bottom panel")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
