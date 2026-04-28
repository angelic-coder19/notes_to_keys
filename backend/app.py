"""
app.py
Notes to Keys — Flask Inference Server (Option A)

In production (Hugging Face Spaces), Flask serves both:
  - The React frontend from frontend/dist/
  - The inference API routes (/transcribe, /health, /model-info)

Routes:
    POST /transcribe    — accepts audio file, returns transcription JSON
    GET  /health        — confirms server is running
    GET  /model-info    — returns config the frontend needs
    GET  /*             — serves the React SPA (index.html + static assets)

Usage (development):
    python app.py
    React dev server runs separately on http://localhost:5173
    Vite proxies /transcribe, /health, /model-info to this server.

Usage (production / Hugging Face Spaces):
    Build React first:  cd frontend && npm run build
    Then:               python app.py
    Everything served from http://localhost:7860
"""

import os
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from inference import TranscriptionPipeline, PREPROC_CFG
from model_utils import MODEL_CFG

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')

# Path to the built React app.
# - Local dev (backend/ folder):       ../frontend/dist
# - Hugging Face Spaces (flat layout):  ./frontend/dist
# Override via FRONTEND_DIST env variable if your layout differs.
_default_dist = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
FRONTEND_DIST = os.environ.get('FRONTEND_DIST', _default_dist)

# Maximum upload size: 50 MB
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Accepted audio extensions
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# CORS — allow Vite dev server during local development.
# In production (Option A), everything is same-origin so CORS is not needed,
# but keeping it here does no harm and eases local development.
CORS_ORIGINS = [
    'http://localhost:5173',
    'http://localhost:3000',
    'http://127.0.0.1:5173',
    # 'https://your-space.hf.space',  ← add when deployed if needed
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

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"\n\nModel file not found: {MODEL_PATH}\n"
        f"Copy best_model.pth into the backend/ folder and restart.\n"
    )

pipeline = TranscriptionPipeline(MODEL_PATH)

print("  Server ready.")
print("=" * 55)
print()

# ============================================================================
# HELPERS
# ============================================================================

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def error_response(message: str, status_code: int):
    return jsonify({'error': message}), status_code


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Confirm the server is running and the model is loaded."""
    return jsonify({
        'status': 'ok',
        'model':  Path(MODEL_PATH).name,
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return the configuration values the frontend needs."""
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


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Accept an audio file, run inference, return the transcription.

    Request:  multipart/form-data, field name 'audio'
    Response: JSON transcription result (see inference.py for schema)
    """
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
# FRONTEND — serve the React SPA (Option A)
# ============================================================================

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """
    Serve the built React app for any route not matched by the API routes above.

    - Existing files in frontend/dist/ are served directly
      (JS bundles, CSS, images, fonts).
    - Everything else falls back to index.html so React handles the path.
    """
    dist = os.path.abspath(FRONTEND_DIST)

    if not os.path.isdir(dist):
        return (
            '<h2>Frontend not built.</h2>'
            '<p>Run <code>cd frontend && npm run build</code> first, '
            'or start the Vite dev server on port 5173.</p>',
            404,
        )

    target = os.path.join(dist, path)
    if path and os.path.isfile(target):
        return send_from_directory(dist, path)

    return send_from_directory(dist, 'index.html')


# ============================================================================
# LARGE FILE ERROR HANDLER
# ============================================================================

@app.errorhandler(413)
def file_too_large(e):
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return error_response(
        f'File exceeds the {limit_mb} MB upload limit. '
        'Trim the audio before uploading.',
        413
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Hugging Face Spaces expects port 7860.
    # Local development uses 5000 (Vite proxies to this).
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)