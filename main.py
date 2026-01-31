import os
import torch
import torchaudio
import pretty_midi
import numpy as np

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000  # How number of audio samples per second
N_MELS = 229         # Matches Onsets & Frames
HOP_LENGTH = 512     # ~32ms hop at 16kHz
N_FFT = 2048         # Good balance for piano harmonics

DATASET_ROOT = "maestro\maestro-v3.0.0\maestro-v3.0.0"
OUTPUT_ROOT = "processed"

os.makedirs(os.path.join(OUTPUT_ROOT, "audio"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "labels"), exist_ok=True)


# ============================================================
# AUDIO PROCESSING
# ============================================================

def load_audio(path):
    """
    Loads audio using torchaudio.
    Converts to mono and normalizes.
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to target SR
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Normalize to [-1, 1]
    waveform = waveform / waveform.abs().max()

    return waveform


def audio_to_logmel(waveform):
    """
    Converts waveform → log-mel spectrogram.
    This is the actual input to the neural network.
    """
    mel = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_spec = mel(waveform)
    logmel = AmplitudeToDB()(mel_spec)

    return logmel.squeeze(0)  # remove channel dim


# ============================================================
# MIDI PROCESSING
# ============================================================

def load_midi(path):
    """
    Loads MIDI using pretty_midi and extracts note events.
    """
    midi = pretty_midi.PrettyMIDI(path)
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            notes.append({
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "velocity": note.velocity
            })

    return notes


def midi_to_frame_labels(notes, num_frames):
    """
    Converts MIDI note events into frame-level labels.
    Produces a piano-roll style matrix: [88 keys, time_frames].
    """
    labels = np.zeros((88, num_frames), dtype=np.float32)

    for note in notes:
        pitch = note["pitch"] - 21  # MIDI 21 = A0 (lowest piano key)
        if 0 <= pitch < 88:
            start_frame = int(note["start"] * SAMPLE_RATE / HOP_LENGTH)
            end_frame = int(note["end"] * SAMPLE_RATE / HOP_LENGTH)

            end_frame = min(end_frame, num_frames - 1)

            labels[pitch, start_frame:end_frame+1] = 1.0

    return torch.tensor(labels)


# ============================================================
# FILE PAIRING LOGIC
# ============================================================

def find_pairs(root):
    """
    Walks through the MAESTRO dataset and finds matching
    .wav and .midi files based on filename.
    """
    pairs = []

    for year_folder in os.listdir(root):
        year_path = os.path.join(root, year_folder)

        if not os.path.isdir(year_path):
            continue

        files = os.listdir(year_path)
        wavs = [f for f in files if f.endswith(".wav")]
        midis = [f for f in files if f.endswith(".midi") or f.endswith(".mid")]

        # Match by base filename
        wav_bases = {os.path.splitext(f)[0]: f for f in wavs}
        midi_bases = {os.path.splitext(f)[0]: f for f in midis}

        for base in wav_bases:
            if base in midi_bases:
                pairs.append((
                    os.path.join(year_path, wav_bases[base]),
                    os.path.join(year_path, midi_bases[base])
                ))

    return pairs


# ============================================================
# MAIN PREPROCESSING LOOP
# ============================================================

def process_dataset():
    pairs = find_pairs(DATASET_ROOT)

    print(f"Found {len(pairs)} audio/MIDI pairs.")

    for wav_path, midi_path in pairs:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"Processing {base}...")

        # -------------------------
        # AUDIO → LOG-MEL
        # -------------------------
        waveform = load_audio(wav_path)
        logmel = audio_to_logmel(waveform)

        # -------------------------
        # MIDI → LABELS
        # -------------------------
        notes = load_midi(midi_path)
        num_frames = logmel.shape[1]
        labels = midi_to_frame_labels(notes, num_frames)

        # -------------------------
        # SAVE OUTPUT
        # -------------------------
        torch.save(logmel, os.path.join(OUTPUT_ROOT, "audio", f"{base}.pt"))
        torch.save(labels, os.path.join(OUTPUT_ROOT, "labels", f"{base}.pt"))

    print("Preprocessing complete.")


if __name__ == "__main__":
    process_dataset()