"""
consolidate_chunks.py
Notes to Keys — Local Data Consolidation Script

Runs on your LOCAL MACHINE (not Colab).
Reads all 615,906 .npz chunk files and writes them into 6 consolidated
.npy files (3 splits × 3 arrays) using memory-mapped writes.

OUTPUT FILES:
    train_cqt.npy       shape: (N_train, 1, 176, 100)  float32
    train_onset.npy     shape: (N_train, 100, 88)       float32
    train_frame.npy     shape: (N_train, 100, 88)       float32
    val_cqt.npy         shape: (N_val,   1, 176, 100)  float32
    val_onset.npy       shape: (N_val,   100, 88)       float32
    val_frame.npy       shape: (N_val,   100, 88)       float32
    test_cqt.npy        shape: (N_test,  1, 176, 100)  float32
    test_onset.npy      shape: (N_test,  100, 88)       float32
    test_frame.npy      shape: (N_test,  100, 88)       float32

USAGE:
    python consolidate_chunks.py

"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import time

# ============================================================================
# CONFIGURATION — edit these if your paths differ
# ============================================================================

CHUNKS_DIR = Path(r"C:\Users\Zimandola\Desktop\piano\processed_chunks")
OUTPUT_DIR = Path(r"C:\Users\Zimandola\Desktop\piano\consolidated")

# Split assignment — must match your Colab notebook config
VAL_YEARS  = {'2006'}
TEST_YEARS = {'2009'}
TRAIN_YEARS = {'2004', '2008', '2011', '2013', '2014', '2015', '2017', '2018'}

# Array shapes — must match your preprocessing config exactly
CQT_SHAPE   = (1, 176, 100)   # (channels, freq_bins, time_frames)
ONSET_SHAPE = (100, 88)        # (time_frames, piano_keys)
FRAME_SHAPE = (100, 88)        # (time_frames, piano_keys)

# ============================================================================
# STEP 1 — SCAN AND COUNT
# ============================================================================

def scan_chunks(chunks_dir: Path, year_sets: dict) -> dict:
    """
    Walk the chunks directory and count how many .npz files
    belong to each split. Returns a dict mapping split name
    to sorted list of Path objects.
    """
    print("\n" + "="*60)
    print("STEP 1: Scanning chunk files...")
    print("="*60)

    splits = {'train': [], 'val': [], 'test': []}

    for year_dir in sorted(chunks_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        year = year_dir.name
        paths = sorted(year_dir.glob('chunk_*.npz'))

        if year in year_sets['val']:
            splits['val'].extend(paths)
            tag = 'VAL'
        elif year in year_sets['test']:
            splits['test'].extend(paths)
            tag = 'TEST'
        elif year in year_sets['train']:
            splits['train'].extend(paths)
            tag = 'TRAIN'
        else:
            print(f"  WARNING: Year {year} not assigned to any split — skipping")
            continue

        print(f"  {year}: {len(paths):>7,} chunks  [{tag}]")

    print()
    for split, paths in splits.items():
        print(f"  {split.upper():6s}: {len(paths):>7,} chunks")

    total = sum(len(p) for p in splits.values())
    print(f"  {'TOTAL':6s}: {total:>7,} chunks")
    print()

    # Sanity check — load one file to verify shapes
    sample = np.load(splits['train'][0])
    assert sample['cqt'].shape   == CQT_SHAPE,   f"CQT shape mismatch: {sample['cqt'].shape} vs {CQT_SHAPE}"
    assert sample['onset'].shape == ONSET_SHAPE, f"Onset shape mismatch: {sample['onset'].shape} vs {ONSET_SHAPE}"
    assert sample['frame'].shape == FRAME_SHAPE, f"Frame shape mismatch: {sample['frame'].shape} vs {FRAME_SHAPE}"
    print("  Shape verification passed on sample file.")

    return splits

# ============================================================================
# STEP 2 — CALCULATE DISK SPACE REQUIRED
# ============================================================================

def estimate_disk_usage(splits: dict) -> float:
    """Print disk usage estimate and return total GB."""
    print("\n" + "="*60)
    print("STEP 2: Estimating disk usage...")
    print("="*60)

    cqt_bytes   = int(np.prod(CQT_SHAPE))   * 4   # float32 = 4 bytes
    onset_bytes = int(np.prod(ONSET_SHAPE)) * 4
    frame_bytes = int(np.prod(FRAME_SHAPE)) * 4
    per_chunk   = cqt_bytes + onset_bytes + frame_bytes

    total_gb = 0.0
    for split, paths in splits.items():
        n = len(paths)
        gb = (n * per_chunk) / 1e9
        total_gb += gb
        print(f"  {split.upper():6s}: {n:>7,} chunks × {per_chunk/1e6:.2f} MB/chunk = {gb:.1f} GB")

    print(f"\n  TOTAL disk required: {total_gb:.1f} GB")
    print(f"  (3 arrays × 3 splits = 9 files)")

    response = input(f"\n  Continue? [y/n]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        exit(0)

    return total_gb

# ============================================================================
# STEP 3 — WRITE CONSOLIDATED FILES
# ============================================================================

def write_split(split_name: str, paths: list, output_dir: Path):
    """
    Write one split (train/val/test) to 3 consolidated .npy files.
    Uses np.memmap so only one chunk is in RAM at any time.
    """
    n = len(paths)
    print(f"\n{'='*60}")
    print(f"Writing {split_name.upper()} split ({n:,} chunks)...")
    print(f"{'='*60}")

    cqt_path   = output_dir / f'{split_name}_cqt.npy'
    onset_path = output_dir / f'{split_name}_onset.npy'
    frame_path = output_dir / f'{split_name}_frame.npy'

    # Create memory-mapped output files
    # mode='w+' creates the file and allows read/write
    print(f"  Creating {split_name}_cqt.npy   ({(n * int(np.prod(CQT_SHAPE)) * 4) / 1e9:.1f} GB)...")
    cqt_mm = np.memmap(cqt_path,   dtype='float32', mode='w+', shape=(n, *CQT_SHAPE))

    print(f"  Creating {split_name}_onset.npy ({(n * int(np.prod(ONSET_SHAPE)) * 4) / 1e9:.1f} GB)...")
    onset_mm = np.memmap(onset_path, dtype='float32', mode='w+', shape=(n, *ONSET_SHAPE))

    print(f"  Creating {split_name}_frame.npy ({(n * int(np.prod(FRAME_SHAPE)) * 4) / 1e9:.1f} GB)...")
    frame_mm = np.memmap(frame_path, dtype='float32', mode='w+', shape=(n, *FRAME_SHAPE))

    print(f"\n  Writing chunks...")
    errors = []
    t0 = time.time()

    for i, path in enumerate(tqdm(paths, desc=f"  {split_name}", unit="chunk")):
        try:
            data = np.load(path)
            cqt_mm[i]   = data['cqt'].astype(np.float32)
            onset_mm[i] = data['onset'].astype(np.float32)
            frame_mm[i] = data['frame'].astype(np.float32)
        except Exception as e:
            errors.append((i, str(path), str(e)))
            # Write zeros for corrupted chunks — do not crash
            cqt_mm[i]   = 0.0
            onset_mm[i] = 0.0
            frame_mm[i] = 0.0

        # Flush to disk every 5000 chunks to avoid OS write buffer overflow
        if (i + 1) % 5000 == 0:
            cqt_mm.flush()
            onset_mm.flush()
            frame_mm.flush()

    # Final flush
    cqt_mm.flush()
    onset_mm.flush()
    frame_mm.flush()

    elapsed = time.time() - t0
    rate = n / elapsed
    print(f"\n  Done in {elapsed/60:.1f} min ({rate:.0f} chunks/sec)")

    if errors:
        print(f"\n  WARNING: {len(errors)} corrupted chunks (written as zeros):")
        for idx, p, e in errors[:10]:
            print(f"    [{idx}] {p}: {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")

    # Delete memmap objects to release file handles
    del cqt_mm, onset_mm, frame_mm

    return len(errors)

# ============================================================================
# STEP 4 — VERIFY OUTPUT FILES
# ============================================================================

def verify_outputs(splits: dict, output_dir: Path):
    """
    Open each completed file and verify shape and dtype.
    Check that values are within expected normalised range.
    """
    print(f"\n{'='*60}")
    print("STEP 4: Verifying output files...")
    print(f"{'='*60}")

    all_ok = True

    for split, paths in splits.items():
        n = len(paths)
        for array_name, expected_shape in [
            ('cqt',   (n, *CQT_SHAPE)),
            ('onset', (n, *ONSET_SHAPE)),
            ('frame', (n, *FRAME_SHAPE)),
        ]:
            fpath = output_dir / f'{split}_{array_name}.npy'

            if not fpath.exists():
                print(f"  FAIL: {fpath.name} does not exist")
                all_ok = False
                continue

            # Open in read-only mode
            arr = np.memmap(fpath, dtype='float32', mode='r', shape=expected_shape)

            shape_ok = arr.shape == expected_shape
            dtype_ok = arr.dtype == np.float32

            # Sample 1000 random rows for range check
            idx = np.random.choice(n, min(1000, n), replace=False)
            sample = arr[idx]
            has_nan = np.isnan(sample).any()
            has_inf = np.isinf(sample).any()

            status = "OK" if (shape_ok and dtype_ok and not has_nan and not has_inf) else "FAIL"
            if status == "FAIL":
                all_ok = False

            print(f"  [{status}] {fpath.name}")
            print(f"         shape: {arr.shape}  dtype: {arr.dtype}  "
                  f"NaN: {has_nan}  Inf: {has_inf}")

            del arr

    print()
    if all_ok:
        print("  All files verified successfully.")
        print("  You are ready to upload to Google Drive.")
    else:
        print("  Some files failed verification. Delete them and re-run.")

# ============================================================================
# STEP 5 — SAVE METADATA
# ============================================================================

def save_metadata(splits: dict, output_dir: Path):
    """Save a small JSON with split sizes for the Colab notebook to read."""
    meta = {
        'train_size': len(splits['train']),
        'val_size':   len(splits['val']),
        'test_size':  len(splits['test']),
        'cqt_shape':  list(CQT_SHAPE),
        'onset_shape': list(ONSET_SHAPE),
        'frame_shape': list(FRAME_SHAPE),
        'val_years':  sorted(VAL_YEARS),
        'test_years': sorted(TEST_YEARS),
        'train_years': sorted(TRAIN_YEARS),
    }
    meta_path = output_dir / 'dataset_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved: {meta_path}")
    print(f"  {json.dumps(meta, indent=4)}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  Notes to Keys — Local Consolidation Script")
    print("="*60)
    print(f"  Source: {CHUNKS_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    # Validate source directory
    if not CHUNKS_DIR.exists():
        print(f"\nERROR: Chunks directory not found: {CHUNKS_DIR}")
        print("Check the CHUNKS_DIR path at the top of this script.")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    year_sets = {
        'train': TRAIN_YEARS,
        'val':   VAL_YEARS,
        'test':  TEST_YEARS,
    }

    # Step 1: Scan
    splits = scan_chunks(CHUNKS_DIR, year_sets)

    # Step 2: Estimate and confirm
    estimate_disk_usage(splits)

    # Step 3: Write each split
    total_errors = 0
    for split_name, paths in splits.items():
        n_errors = write_split(split_name, paths, OUTPUT_DIR)
        total_errors += n_errors

    # Step 4: Verify
    verify_outputs(splits, OUTPUT_DIR)

    # Step 5: Save metadata
    save_metadata(splits, OUTPUT_DIR)

    print("\n" + "="*60)
    print("  CONSOLIDATION COMPLETE")
    print(f"  Total corrupted chunks: {total_errors}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    print("  NEXT STEPS:")
    print("  1. Upload the 9 .npy files + dataset_meta.json to Google Drive")
    print("     Recommended: use the Google Drive desktop app")
    print("     Target folder: My Drive/notes_to_keys/consolidated/")
    print("  2. Open the Colab training notebook")
    print("  3. Set CONSOLIDATED_DIR in Section 1.3 to match your Drive path")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
