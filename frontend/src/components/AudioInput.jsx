import { useState, useCallback } from 'react';

const ACCEPTED = '.wav,.mp3,.flac,.ogg,.m4a';

/**
 * AudioInput
 *
 * Drag-and-drop + click-to-browse upload area.
 * Calls onFile(file) when a valid audio file is selected.
 */
export default function AudioInput({ onFile, disabled = false }) {
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback((file) => {
    if (!file) return;
    onFile(file);
  }, [onFile]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile, disabled]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  }, [disabled]);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
    e.target.value = '';
  }, [handleFile]);

  return (
    <div className="upload-page">
      <h1 className="upload-hero">
        Upload your piano recording.<br />
        <span>See every key it played.</span>
      </h1>

      <p className="upload-sub">
        A neural network trained on 615,906 samples will transcribe 
        your recording into a colour-coded piano roll in seconds.
      </p>

      <div
        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        aria-label="Upload audio file"
      >
        <input
          type="file"
          accept={ACCEPTED}
          onChange={onChange}
          disabled={disabled}
          aria-label="Choose audio file"
        />

        {/* Piano key decorative strip */}
        <PianoStrip />

        <div className="upload-zone-icon">🎹</div>
        <div className="upload-zone-title">
          {dragOver ? 'Drop it here' : 'Drop your audio file here'}
        </div>
        <div className="upload-zone-sub">or click anywhere to browse</div>
        <div className="upload-zone-formats">WAV · MP3 · FLAC · OGG · M4A · max 50 MB</div>
      </div>

      <TryItNote />
    </div>
  );
}

/**
 * Small decorative piano strip shown in the upload zone.
 */
function PianoStrip() {
  // Mini keyboard: 2 octaves of white keys with black keys
  const WHITE_W = 22;
  const WHITE_H = 40;
  const BLACK_W = 14;
  const BLACK_H = 26;
  const N_WHITE = 14;
  const COLORS = ['#EF5350','#FF7043','#FFA726','#66BB6A',
                   '#26C6DA','#5C6BC0','#AB47BC','#EC407A'];

  // 14 white keys, 10 black keys for a nice visual
  const blackPositions = [0.65,1.65,3.65,4.65,5.65,7.65,8.65,10.65,11.65,12.65];

  return (
    <svg
      width={N_WHITE * WHITE_W}
      height={WHITE_H + 4}
      viewBox={`0 0 ${N_WHITE * WHITE_W} ${WHITE_H + 4}`}
      style={{ borderRadius: 6, overflow: 'hidden' }}
      aria-hidden="true"
    >
      {/* White keys */}
      {Array.from({ length: N_WHITE }, (_, i) => (
        <rect
          key={i}
          x={i * WHITE_W}
          y={4}
          width={WHITE_W - 1}
          height={WHITE_H}
          fill="#FFFFFF"
          stroke="#E0E0E0"
          strokeWidth={0.5}
          rx={3}
        />
      ))}
      {/* Black keys with octave colors */}
      {blackPositions.map((pos, i) => (
        <rect
          key={i}
          x={pos * WHITE_W + (WHITE_W - BLACK_W) / 2}
          y={4}
          width={BLACK_W}
          height={BLACK_H}
          fill={COLORS[i % COLORS.length]}
          rx={3}
        />
      ))}
    </svg>
  );
}

/**
 * A small friendly hint about what kinds of recordings work well.
 */
function TryItNote() {
  return (
    <p style={{
      fontSize: 12,
      color: 'var(--text-muted)',
      maxWidth: 380,
      lineHeight: 1.7,
      textAlign: 'center',
    }}>
      Works best with solo piano recordings. MIDI-recorded audio (like MAESTRO) 
      gives the clearest results. Live recordings work too.
    </p>
  );
}
