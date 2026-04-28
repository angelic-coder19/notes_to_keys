import { useMemo } from 'react';
import { buildKeyLayout, getNoteColor } from '../utils/noteUtils.js';

const WHITE_KEY_W  = 18;   // base width — SVG viewBox scales to container
const WHITE_KEY_H  = 68;
const BLACK_KEY_W  = 11;
const BLACK_KEY_H  = 44;
const BLACK_OFFSET = (WHITE_KEY_W - BLACK_KEY_W) / 2;   // center black over edge

// Pre-compute layout once at module level (doesn't change)
const { keys, totalWhiteKeys } = buildKeyLayout();
const SVG_WIDTH  = totalWhiteKeys * WHITE_KEY_W;        // 52 × 18 = 936
const SVG_HEIGHT = WHITE_KEY_H + 4;

/**
 * PianoKeyboard
 *
 * Renders an SVG 88-key piano keyboard.
 * Keys in activeNotes light up in their octave color.
 *
 * Props:
 *   activeNotes  {Set<number>} - set of currently active MIDI pitches
 */
export default function PianoKeyboard({ activeNotes = new Set() }) {
  // Split into white and black key groups so black keys render on top
  const { whiteKeys, blackKeys } = useMemo(() => {
    const white = keys.filter(k => !k.isBlack);
    const black = keys.filter(k => k.isBlack);
    return { whiteKeys: white, blackKeys: black };
  }, []);

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
      aria-label="Piano keyboard showing active notes"
      role="img"
      style={{ display: 'block' }}
    >
      <title>Piano keyboard</title>

      {/* White keys (rendered first, black keys overlay on top) */}
      {whiteKeys.map(key => {
        const isActive = activeNotes.has(key.pitch);
        const x        = key.x * WHITE_KEY_W;

        return (
          <g key={key.pitch}>
            <rect
              x={x + 0.5}
              y={2}
              width={WHITE_KEY_W - 1}
              height={WHITE_KEY_H}
              rx={3}
              fill={isActive ? getNoteColor(key.pitch) : '#FFFFFF'}
              stroke={isActive ? getNoteColor(key.pitch) : '#D0D0DC'}
              strokeWidth={0.5}
              style={{ transition: 'fill 0.06s, stroke 0.06s' }}
            />
            {/* Highlight strip at top of active white key */}
            {isActive && (
              <rect
                x={x + 0.5}
                y={2}
                width={WHITE_KEY_W - 1}
                height={6}
                rx={3}
                fill="rgba(255,255,255,0.45)"
                style={{ pointerEvents: 'none' }}
              />
            )}
          </g>
        );
      })}

      {/* Black keys (rendered on top) */}
      {blackKeys.map(key => {
        const isActive = activeNotes.has(key.pitch);
        // key.x is in white-key-width units; center over the white key boundary
        const x        = key.x * WHITE_KEY_W - BLACK_KEY_W / 2 + WHITE_KEY_W / 2;

        return (
          <rect
            key={key.pitch}
            x={x}
            y={2}
            width={BLACK_KEY_W}
            height={BLACK_KEY_H}
            rx={3}
            fill={isActive ? getNoteColor(key.pitch) : '#1A1A2E'}
            style={{ transition: 'fill 0.06s' }}
          />
        );
      })}

      {/* Middle C label — C4 is MIDI 60, key index 39 */}
      <MiddleCLabel whiteKeyWidth={WHITE_KEY_W} keyMap={whiteKeys} />
    </svg>
  );
}

/**
 * Small "C4" label under middle C.
 */
function MiddleCLabel({ whiteKeyWidth, keyMap }) {
  const c4 = keyMap.find(k => k.pitch === 60);
  if (!c4) return null;

  const x = c4.x * whiteKeyWidth + whiteKeyWidth / 2;
  const y = WHITE_KEY_H - 4;

  return (
    <text
      x={x}
      y={y + 2}
      textAnchor="middle"
      fill="#AAAACC"
      fontSize="8"
      fontFamily="Inter, sans-serif"
      textLength="12"
    >
      C4
    </text>
  );
}
