import { formatTime } from '../utils/noteUtils.js';

/**
 * PlaybackControls
 *
 * Play/pause button, scrubber, and time display.
 * Sits in the bar between the piano roll and keyboard.
 *
 * Props:
 *   isPlaying   {boolean}
 *   currentTime {number}  seconds
 *   duration    {number}  seconds
 *   onToggle    {fn}
 *   onSeek      {fn(time)}
 */
export default function PlaybackControls({
  isPlaying,
  currentTime,
  duration,
  onToggle,
  onSeek,
}) {
  const pct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="playback-bar">

      {/* Play / Pause */}
      <button
        className="btn btn-primary btn-icon"
        onClick={onToggle}
        aria-label={isPlaying ? 'Pause' : 'Play'}
        title={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? '⏸' : '▶'}
      </button>

      {/* Seek bar */}
      <input
        type="range"
        className="seek-bar"
        min={0}
        max={duration || 0}
        step={0.1}
        value={currentTime}
        onChange={e => onSeek(parseFloat(e.target.value))}
        aria-label="Seek"
        style={{
          background: `linear-gradient(to right, var(--accent) ${pct}%, var(--border) ${pct}%)`,
        }}
      />

      {/* Time display */}
      <div className="time-display" aria-live="off">
        {formatTime(currentTime)}
        <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
          {' / '}{formatTime(duration)}
        </span>
      </div>

    </div>
  );
}
