// ── OCTAVE COLORS ──────────────────────────────────────────────────────────
// Eight distinct colors, one per piano octave.
// Warm at the low end, cool in the middle, back to warm at the top.
// All are Material Design 400-level — vibrant on white backgrounds.
export const OCTAVE_COLORS = [
  '#EF5350', // octave 0  (A0 – B0)   : vivid red
  '#FF7043', // octave 1  (C1 – B1)   : deep orange
  '#FFA726', // octave 2  (C2 – B2)   : amber
  '#66BB6A', // octave 3  (C3 – B3)   : green
  '#26C6DA', // octave 4  (C4 – B4)   : cyan   ← middle C lives here
  '#5C6BC0', // octave 5  (C5 – B5)   : indigo
  '#AB47BC', // octave 6  (C6 – B6)   : purple
  '#EC407A', // octave 7  (C7 – C8)   : pink
];

// ── PITCH HELPERS ───────────────────────────────────────────────────────────

/**
 * Return the display color for a MIDI pitch (21–108).
 */
export function getNoteColor(pitch) {
  // MIDI octave convention: C4 = 60, so octave = floor(pitch / 12) - 1
  const octave = Math.floor(pitch / 12) - 1;
  return OCTAVE_COLORS[Math.min(Math.max(octave, 0), 7)];
}

/**
 * Return the note name string for a MIDI pitch (e.g. "C4", "A#3").
 */
export function getNoteName(pitch) {
  const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(pitch / 12) - 1;
  return `${NAMES[pitch % 12]}${octave}`;
}

/**
 * Return true if a MIDI pitch corresponds to a black piano key.
 */
export function isBlackKey(pitch) {
  return [1, 3, 6, 8, 10].includes(pitch % 12);
}

// ── TIME HELPERS ────────────────────────────────────────────────────────────

/**
 * Format seconds as M:SS (e.g. 78.9 → "1:18").
 */
export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// ── ACTIVE NOTE LOOKUP ──────────────────────────────────────────────────────

/**
 * Return the set of MIDI pitches that are active (being held) at currentTime.
 * Used to light up keyboard keys during playback.
 *
 * @param {Array}  notes       - transcription note list from the API
 * @param {number} currentTime - current playback position in seconds
 * @returns {Set<number>}       - set of active MIDI pitches
 */
export function getActiveNotes(notes, currentTime) {
  const active = new Set();
  for (const note of notes) {
    if (note.start_time <= currentTime && note.end_time > currentTime) {
      active.add(note.pitch);
    }
  }
  return active;
}

// ── KEYBOARD LAYOUT ─────────────────────────────────────────────────────────

/**
 * Pre-compute the position and type (white/black) of all 88 piano keys.
 *
 * Returns:
 *   keys: array of { pitch, isBlack, x } where x is the left edge
 *          in units of whiteKeyWidth.
 *   totalWhiteKeys: number of white keys (52 for a full 88-key piano)
 *
 * The caller multiplies x by the actual white key width in pixels to get
 * screen coordinates. Black key x values are already offset to sit between
 * their neighboring white keys.
 */
export function buildKeyLayout() {
  const keys = [];
  let whiteIndex = 0;

  for (let pitch = 21; pitch <= 108; pitch++) {
    const black = isBlackKey(pitch);

    if (black) {
      // Sit the black key between whiteIndex-1 and whiteIndex.
      // x = whiteIndex * 1 unit, minus half a black key width.
      // We store x as a fraction of whiteKeyWidth; caller scales it.
      keys.push({ pitch, isBlack: true, x: whiteIndex - 0.5 });
    } else {
      keys.push({ pitch, isBlack: false, x: whiteIndex });
      whiteIndex++;
    }
  }

  return { keys, totalWhiteKeys: whiteIndex }; // totalWhiteKeys === 52
}
