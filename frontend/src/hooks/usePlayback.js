import { useState, useRef, useEffect, useCallback } from 'react';

/**
 * usePlayback
 *
 * Manages the HTML5 audio element and exposes playback controls.
 * The audio ref is attached to a hidden <audio> element rendered in App.
 *
 * Usage:
 *   const pb = usePlayback(audioUrl);
 *   <audio ref={pb.audioRef} />
 *   pb.toggle()
 *   pb.seek(30.5)
 */
export function usePlayback(audioUrl) {
  const audioRef                  = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration]   = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // When a new audio URL is set, load it and reset state
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;

    audio.src = audioUrl;
    audio.load();
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
  }, [audioUrl]);

  // Attach event listeners to the audio element
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTimeUpdate    = () => setCurrentTime(audio.currentTime);
    const onDuration      = () => setDuration(isFinite(audio.duration) ? audio.duration : 0);
    const onPlay          = () => setIsPlaying(true);
    const onPause         = () => setIsPlaying(false);
    const onEnded         = () => { setIsPlaying(false); setCurrentTime(0); };
    const onSeeked        = () => setCurrentTime(audio.currentTime);

    audio.addEventListener('timeupdate',    onTimeUpdate);
    audio.addEventListener('durationchange',onDuration);
    audio.addEventListener('loadedmetadata',onDuration);
    audio.addEventListener('play',          onPlay);
    audio.addEventListener('pause',         onPause);
    audio.addEventListener('ended',         onEnded);
    audio.addEventListener('seeked',        onSeeked);

    return () => {
      audio.removeEventListener('timeupdate',    onTimeUpdate);
      audio.removeEventListener('durationchange',onDuration);
      audio.removeEventListener('loadedmetadata',onDuration);
      audio.removeEventListener('play',          onPlay);
      audio.removeEventListener('pause',         onPause);
      audio.removeEventListener('ended',         onEnded);
      audio.removeEventListener('seeked',        onSeeked);
    };
  }, []);

  const play   = useCallback(() => audioRef.current?.play(), []);
  const pause  = useCallback(() => audioRef.current?.pause(), []);
  const toggle = useCallback(() => {
    if (!audioRef.current) return;
    if (audioRef.current.paused) audioRef.current.play();
    else audioRef.current.pause();
  }, []);

  const seek = useCallback((time) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, time);
    setCurrentTime(Math.max(0, time));
  }, []);

  return { audioRef, currentTime, duration, isPlaying, play, pause, toggle, seek };
}
