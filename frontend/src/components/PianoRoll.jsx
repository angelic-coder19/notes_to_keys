import { useRef, useEffect, useCallback } from 'react';
import { getNoteColor } from '../utils/noteUtils.js';

// ── LAYOUT CONSTANTS ────────────────────────────────────────────────────────
const PX_PER_SEC  = 100;   // horizontal pixels per second of audio
const KEY_H       = 8;     // pixel height of each piano key row
const N_KEYS      = 88;
const LEFT_MARGIN = 56;    // space for pitch labels (e.g. "C4")
const TOP_PAD     = 8;
const BOT_PAD     = 20;    // space for time labels
const CANVAS_H    = N_KEYS * KEY_H + TOP_PAD + BOT_PAD;

// C notes on the 88-key piano — used for grid lines and labels
// MIDI pitches for C1 through C8
const C_PITCHES = [24, 36, 48, 60, 72, 84, 96, 108];

/**
 * Convert a MIDI pitch (21–108) to a Y pixel coordinate.
 * High pitches are at the TOP of the canvas (standard piano roll convention).
 */
function pitchToY(pitch) {
  const keyIndex = pitch - 21;                    // 0 = A0, 87 = C8
  return TOP_PAD + (N_KEYS - 1 - keyIndex) * KEY_H;
}

/**
 * PianoRoll
 *
 * Renders the transcription as a colour-coded piano roll on an HTML Canvas.
 * A CSS-positioned playhead line moves without redrawing the canvas, keeping
 * playback smooth even for long recordings.
 *
 * Props:
 *   notes       {Array}  - transcription note list from the API
 *   duration    {number} - total audio duration in seconds
 *   currentTime {number} - current playback position in seconds
 */
export default function PianoRoll({ notes, duration, currentTime }) {
  const canvasRef    = useRef(null);
  const containerRef = useRef(null);

  const canvasWidth = Math.max(LEFT_MARGIN + duration * PX_PER_SEC + 24, 900);

  // ── DRAW ─────────────────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Clear with white background
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Alternate octave bands for subtle depth
    for (let oct = 0; oct < 8; oct++) {
      const lowPitch  = oct === 0 ? 21 : 12 * (oct + 1);
      const highPitch = Math.min(12 * (oct + 2) - 1, 108);
      if (lowPitch > 108) break;

      const yTop    = pitchToY(highPitch);
      const yBottom = pitchToY(lowPitch) + KEY_H;
      if (oct % 2 === 0) {
        ctx.fillStyle = '#FAFAFA';
        ctx.fillRect(LEFT_MARGIN, yTop, canvas.width - LEFT_MARGIN, yBottom - yTop);
      }
    }

    // Vertical grid: one line per second, label every 5s
    ctx.save();
    for (let t = 0; t <= Math.ceil(duration); t++) {
      const x = LEFT_MARGIN + t * PX_PER_SEC;
      ctx.strokeStyle = t % 5 === 0 ? '#DDDDE8' : '#EEEEF4';
      ctx.lineWidth   = t % 5 === 0 ? 1 : 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, CANVAS_H - BOT_PAD);
      ctx.stroke();

      if (t % 5 === 0) {
        const m = Math.floor(t / 60);
        const s = t % 60;
        const label = m > 0 ? `${m}:${s.toString().padStart(2, '0')}` : `${t}s`;
        ctx.fillStyle = '#B0B0C8';
        ctx.font      = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, x, CANVAS_H - 4);
      }
    }
    ctx.restore();

    // Horizontal grid: C note lines across each octave boundary
    ctx.save();
    C_PITCHES.forEach(pitch => {
      if (pitch < 21 || pitch > 108) return;
      const y = pitchToY(pitch);
      ctx.strokeStyle = '#DDDDE8';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();

      const octave = Math.floor(pitch / 12) - 1;
      ctx.fillStyle  = '#A0A0B8';
      ctx.font       = '10px Inter, sans-serif';
      ctx.textAlign  = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`C${octave}`, LEFT_MARGIN - 6, y + KEY_H / 2);
    });
    ctx.restore();

    // Left margin divider
    ctx.strokeStyle = '#E0E0EC';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(LEFT_MARGIN, 0);
    ctx.lineTo(LEFT_MARGIN, CANVAS_H);
    ctx.stroke();

    // A0 label
    ctx.fillStyle    = '#A0A0B8';
    ctx.font         = '10px Inter, sans-serif';
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('A0', LEFT_MARGIN - 6, pitchToY(21) + KEY_H / 2);

    // Notes
    if (notes && notes.length > 0) {
      notes.forEach(note => {
        const x = LEFT_MARGIN + note.start_time * PX_PER_SEC;
        const w = Math.max(note.duration * PX_PER_SEC, 3);
        const y = pitchToY(note.pitch);
        const h = KEY_H - 1;

        const color = getNoteColor(note.pitch);

        // Main note rectangle
        ctx.fillStyle = color;
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(x, y, w, h, Math.min(3, w / 2));
        } else {
          ctx.rect(x, y, w, h);
        }
        ctx.fill();

        // Top highlight stripe for depth
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(x, y, w, Math.min(2, h / 2), [Math.min(3, w / 2), Math.min(3, w / 2), 0, 0]);
        } else {
          ctx.rect(x, y, w, 2);
        }
        ctx.fill();
      });
    }

  }, [notes, duration]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Auto-scroll to keep playhead visible (smooth lead)
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !duration) return;

    const playheadX    = LEFT_MARGIN + currentTime * PX_PER_SEC;
    const containerW   = container.clientWidth;
    const scrollLeft   = container.scrollLeft;
    const rightEdge    = scrollLeft + containerW;

    // Scroll when playhead is 70% across the visible window
    if (playheadX > scrollLeft + containerW * 0.70) {
      container.scrollLeft = playheadX - containerW * 0.30;
    }

    // Don't let the playhead go out of view to the left
    if (playheadX < scrollLeft + 20) {
      container.scrollLeft = Math.max(0, playheadX - containerW * 0.30);
    }
  }, [currentTime, duration]);

  const playheadX = LEFT_MARGIN + currentTime * PX_PER_SEC;

  return (
    <div
      ref={containerRef}
      style={{
        overflowX: 'auto',
        overflowY: 'hidden',
        position: 'relative',
        cursor: 'default',
      }}
      role="img"
      aria-label={`Piano roll showing ${notes?.length ?? 0} detected notes over ${Math.round(duration)} seconds`}
    >
      <div style={{ position: 'relative', width: canvasWidth, height: CANVAS_H }}>
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={CANVAS_H}
          style={{ display: 'block' }}
        />

        {/* Playhead — CSS-positioned, no canvas redraw needed */}
        <div
          aria-hidden="true"
          style={{
            position:    'absolute',
            top:         0,
            left:        playheadX,
            width:       2,
            height:      CANVAS_H - BOT_PAD,
            background:  '#FF4757',
            transform:   'translateX(-1px)',
            pointerEvents: 'none',
            boxShadow:   '0 0 8px rgba(255,71,87,0.45)',
            zIndex:      10,
            transition:  'left 0.05s linear',
          }}
        />

        {/* Empty state overlay */}
        {(!notes || notes.length === 0) && (
          <div style={{
            position:    'absolute',
            inset:       0,
            display:     'flex',
            alignItems:  'center',
            justifyContent: 'center',
            color:       'var(--text-muted)',
            fontSize:    14,
            pointerEvents: 'none',
          }}>
            No notes detected
          </div>
        )}
      </div>
    </div>
  );
}
