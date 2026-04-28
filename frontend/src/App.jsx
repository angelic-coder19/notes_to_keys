import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import AudioInput       from './components/AudioInput.jsx';
import PianoRoll        from './components/PianoRoll.jsx';
import PianoKeyboard    from './components/PianoKeyboard.jsx';
import PlaybackControls from './components/PlaybackControls.jsx';
import { useTranscription } from './hooks/useTranscription.js';
import { usePlayback }      from './hooks/usePlayback.js';
import { getActiveNotes, OCTAVE_COLORS, formatTime } from './utils/noteUtils.js';

// ── OCTAVE LEGEND DATA ───────────────────────────────────────────────────────
const LEGEND = [
  { label: 'A0–B0', color: OCTAVE_COLORS[0] },
  { label: 'C1–B1', color: OCTAVE_COLORS[1] },
  { label: 'C2–B2', color: OCTAVE_COLORS[2] },
  { label: 'C3–B3', color: OCTAVE_COLORS[3] },
  { label: 'C4–B4', color: OCTAVE_COLORS[4] },
  { label: 'C5–B5', color: OCTAVE_COLORS[5] },
  { label: 'C6–B6', color: OCTAVE_COLORS[6] },
  { label: 'C7–C8', color: OCTAVE_COLORS[7] },
];

export default function App() {
  const [fileName,  setFileName]  = useState(null);
  const [audioUrl,  setAudioUrl]  = useState(null);
  const prevUrlRef                = useRef(null);

  const { transcribe, result, status, error, reset } = useTranscription();
  const pb = usePlayback(audioUrl);

  // Active notes at current playback position (for keyboard highlighting)
  const activeNotes = useMemo(() => {
    if (!result?.notes) return new Set();
    return getActiveNotes(result.notes, pb.currentTime);
  }, [result, pb.currentTime]);

  // ── FILE HANDLER ──────────────────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    // Revoke previous object URL to avoid memory leak
    if (prevUrlRef.current) {
      URL.revokeObjectURL(prevUrlRef.current);
    }

    const url = URL.createObjectURL(file);
    prevUrlRef.current = url;

    setFileName(file.name);
    setAudioUrl(url);
    transcribe(file);
  }, [transcribe]);

  // Revoke object URL on unmount
  useEffect(() => {
    return () => {
      if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    };
  }, []);

  // ── RESET ─────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    pb.pause();
    setFileName(null);
    setAudioUrl(null);
    reset();
  }, [pb, reset]);

  // ── RENDER ────────────────────────────────────────────────────────────────
  return (
    <>
      <Header />

      <main className="main">

        {/* ── IDLE: show upload ── */}
        {status === 'idle' && (
          <AudioInput onFile={handleFile} />
        )}

        {/* ── PROCESSING ── */}
        {status === 'processing' && (
          <div className="processing-page">
            <div className="processing-spinner">
              <span /><span /><span /><span />
            </div>
            <div className="processing-title">Transcribing your music…</div>
            <div className="processing-file">{fileName}</div>
            <div className="processing-note">
              The neural network is identifying which keys were pressed and when.
              This takes a few seconds for short recordings and up to a minute
              for longer pieces.
            </div>
          </div>
        )}

        {/* ── ERROR ── */}
        {status === 'error' && (
          <div className="error-page">
            <div className="error-icon">⚠️</div>
            <div className="error-title">Transcription failed</div>
            <div className="error-message">{error}</div>
            <button className="btn btn-primary" onClick={handleReset}>
              Try another file
            </button>
          </div>
        )}

        {/* ── DONE: show results ── */}
        {status === 'done' && result && (
          <div className="result-view">

            {/* Stats bar */}
            <div className="stats-bar">
              <div className="stat-chip">
                <span className="stat-icon">🎵</span>
                <strong>{result.n_notes.toLocaleString()}</strong>
                <span className="stat-label">notes</span>
              </div>
              <div className="stat-chip">
                <span className="stat-icon">⏱</span>
                <strong>{formatTime(result.duration_sec)}</strong>
                <span className="stat-label">duration</span>
              </div>
              <div className="stat-chip">
                <span className="stat-icon">⚡</span>
                <strong>{result.inference_time_sec.toFixed(1)}s</strong>
                <span className="stat-label">inference</span>
              </div>
              <div className="stat-chip">
                <span className="stat-icon">🎹</span>
                <span className="stat-label" style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {fileName}
                </span>
              </div>

              <button className="btn btn-secondary btn-new" onClick={handleReset}>
                ↑ New recording
              </button>
            </div>

            {/* Piano roll card */}
            <div className="roll-card">
              <div className="roll-header">
                <span className="roll-title">Piano Roll</span>

                {/* Octave colour legend */}
                <div className="octave-legend">
                  {LEGEND.map(({ label, color }) => (
                    <div key={label} className="legend-label">
                      <div
                        className="legend-dot"
                        style={{ background: color }}
                        title={label}
                      />
                    </div>
                  ))}
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 4 }}>
                    Color = octave
                  </span>
                </div>
              </div>

              <PianoRoll
                notes={result.notes}
                duration={result.duration_sec}
                currentTime={pb.currentTime}
              />

              <PlaybackControls
                isPlaying={pb.isPlaying}
                currentTime={pb.currentTime}
                duration={pb.duration || result.duration_sec}
                onToggle={pb.toggle}
                onSeek={pb.seek}
              />

              <div className="keyboard-wrapper">
                <PianoKeyboard activeNotes={activeNotes} />
              </div>
            </div>

          </div>
        )}
      </main>

      {/* Hidden audio element — controlled by usePlayback */}
      <audio ref={pb.audioRef} preload="auto" style={{ display: 'none' }} />

      <Footer />
    </>
  );
}

// ── HEADER ───────────────────────────────────────────────────────────────────
function Header() {
  return (
    <header className="header">
      <div className="header-logo">
        <div className="header-logo-icon" aria-hidden="true">🎹</div>
        <span className="header-logo-text">
          Notes<span> to Keys</span>
        </span>
      </div>
      <div className="header-tag">AI Transcription</div>
    </header>
  );
}

// ── FOOTER ───────────────────────────────────────────────────────────────────
function Footer() {
  return (
    <footer className="footer">
      Built with an Onsets &amp; Frames neural network trained on the{' '}
      <a href="https://magenta.tensorflow.org/datasets/maestro" target="_blank" rel="noopener noreferrer">
        MAESTRO dataset
      </a>
      {' '}· 615,906 training samples · Onset F1: 0.85
    </footer>
  );
}
