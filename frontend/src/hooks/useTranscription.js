import { useState, useCallback } from 'react';

/**
 * useTranscription
 *
 * Manages the full transcription lifecycle:
 *   idle → processing → done | error
 *
 * Usage:
 *   const { transcribe, result, status, error, reset } = useTranscription();
 *   await transcribe(file);   // File object from input or drop
 */
export function useTranscription() {
  const [result, setResult]   = useState(null);
  const [status, setStatus]   = useState('idle');   // 'idle' | 'processing' | 'done' | 'error'
  const [error, setError]     = useState(null);

  const transcribe = useCallback(async (file) => {
    setStatus('processing');
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await fetch('/transcribe', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server error ${response.status}`);
      }

      setResult(data);
      setStatus('done');

    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setStatus('idle');
    setError(null);
  }, []);

  return { transcribe, result, status, error, reset };
}
