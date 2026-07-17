"""
app.py
Notes to Keys — Flask Inference Server (Option A)

Routes:
    POST /transcribe    — accepts audio file, returns transcription JSON
    GET  /sample        — serves the built-in demo WAV file
    GET  /health        — confirms server is running
    GET  /model-info    — returns config the frontend needs
    GET  /*             — serves the React SPA
"""

import os
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from inference import TranscriptionPipeline, PREPROC_CFG
from model_utils import MODEL_CFG

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')

_default_dist  = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
FRONTEND_DIST  = os.environ.get('FRONTEND_DIST', _default_dist)

# The demo sample WAV — place a short piano recording named sample.wav
# in the same folder as app.py (the backend/ folder / container root).
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'sample.wav')

MAX_UPLOAD_BYTES   = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

CORS_ORIGINS = [
    'http://localhost:5173',
    'http://localhost:3000',
    'http://127.0.0.1:5173',
]

# ============================================================================
# APP INITIALISATION
# ============================================================================

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

CORS(app, origins=CORS_ORIGINS)

print()
print("=" * 55)
print("  Notes to Keys — Inference Server")
print("=" * 55)
print(f"  Model path:    {MODEL_PATH}")
print(f"  Frontend dist: {os.path.abspath(FRONTEND_DIST)}")
print(f"  Sample file:   {SAMPLE_PATH} ({'found' if Path(SAMPLE_PATH).exists() else 'NOT FOUND — demo button will error'})")

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"\n\nModel file not found: {MODEL_PATH}\n"
        f"Copy best_model.pth into the backend/ folder and restart.\n"
    )

pipeline = TranscriptionPipeline(MODEL_PATH)

print("  Server ready.")
print("=" * 55)
print()

# ====================================================
# HELPERS
# ====================================================

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def error_response(message: str, status_code: int):
    return jsonify({'error': message}), status_code


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model':  Path(MODEL_PATH).name,
        'sample': Path(SAMPLE_PATH).exists(),
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'frame_duration_sec':     round(PREPROC_CFG['hop_length'] / PREPROC_CFG['sample_rate'], 6),
        'n_keys':                 MODEL_CFG['n_keys'],
        'sample_rate':            PREPROC_CFG['sample_rate'],
        'hop_length':             PREPROC_CFG['hop_length'],
        'n_freq_bins':            PREPROC_CFG['n_bins'],
        'onset_threshold':        MODEL_CFG['onset_threshold'],
        'frame_threshold':        MODEL_CFG['frame_threshold'],
        'max_audio_duration_sec': 600,
        'max_upload_mb':          MAX_UPLOAD_BYTES // (1024 * 1024),
        'allowed_formats':        sorted(ALLOWED_EXTENSIONS),
    })


@app.route('/sample', methods=['GET'])
def sample():
    """
    Serve the built-in demo WAV file so users can try the app immediately.

    To add the sample:
      1. Pick a short piano recording (30–90 seconds works well).
      2. Name it sample.wav.
      3. Place it in the same folder as app.py (the backend/ folder).
      4. Commit it to the Hugging Face Space repo (it's just a file — no LFS needed
         for files under ~50 MB; use LFS if it's larger).
    """
    if not Path(SAMPLE_PATH).exists():
        return error_response(
            'Demo sample not found on the server. '
            'Add a file named sample.wav to the backend folder.',
            404
        )
    return send_file(SAMPLE_PATH, mimetype='audio/wav', as_attachment=False)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return error_response(
            'No audio file attached. '
            'Send a multipart/form-data request with field name "audio".',
            400
        )

    file = request.files['audio']

    if not file.filename:
        return error_response('File has no name.', 400)

    if not allowed_file(file.filename):
        ext = Path(file.filename).suffix.lower()
        return error_response(
            f'Unsupported format: "{ext}". '
            f'Accepted: {", ".join(sorted(ALLOWED_EXTENSIONS))}.',
            400
        )

    suffix   = Path(file.filename).suffix.lower()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"[/transcribe] {file.filename}  ({size_mb:.1f} MB)")

        result = pipeline.transcribe(tmp_path)

        print(f"[/transcribe] {result['n_notes']} notes  "
              f"{result['duration_sec']:.1f}s audio  "
              f"{result['inference_time_sec']:.2f}s inference")

        return jsonify(result), 200

    except ValueError as e:
        return error_response(str(e), 400)

    except Exception as e:
        print(f"[/transcribe] ERROR: {e}")
        traceback.print_exc()
        return error_response(f'Inference failed: {str(e)}', 500)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================================
# FRONTEND — React SPA (Option A)
# ============================================================================

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    dist   = os.path.abspath(FRONTEND_DIST)

    if not os.path.isdir(dist):
        return (
            '<h2>Frontend not built.</h2>'
            '<p>Run <code>cd frontend && npm run build</code> first.</p>',
            404,
        )

    target = os.path.join(dist, path)
    if path and os.path.isfile(target):
        return send_from_directory(dist, path)

    return send_from_directory(dist, 'index.html')


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def file_too_large(e):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return error_response(
        f'File exceeds the {limit_mb} MB upload limit.',
        413
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
