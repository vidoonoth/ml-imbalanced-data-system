#!/usr/bin/env python3
"""Flask API for the SCARF normal/malware detection model."""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from detection_service import DetectionService
except ImportError:
    DetectionService = None

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_local_env(env_path):
    """Load simple KEY=VALUE pairs when cPanel env vars are not available."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()

        if not line or line.startswith('#') or '=' not in line:
            continue

        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


load_local_env(ROOT_DIR / '.env')

API_KEY = os.getenv('ML_API_KEY', 'ml_api_key_secret_2026')
API_VERSION = '2.0'

detection_service = None


def verify_api_key():
    """Verify API key from request."""
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Bearer '):
        return False
    
    token = auth_header.replace('Bearer ', '').strip()
    return token == API_KEY


def get_detection_service():
    """Get or initialize detection service."""
    global detection_service
    
    if detection_service is None:
        if DetectionService is None:
            raise RuntimeError("DetectionService not available. Check detection_service.py")
        
        detection_service = DetectionService()
    
    return detection_service


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'service': 'ML Detection API',
        'version': API_VERSION,
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'detect_batch': '/api/detect/batch (POST)',
            'detect_single': '/api/detect/single (POST)',
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        service = get_detection_service()
        model_loaded = service.is_ready()
        
        return jsonify({
            'status': 'healthy' if model_loaded else 'initializing',
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/detect/single', methods=['POST'])
def detect_single():
    """Detect single network record."""
    if not verify_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json(silent=True)
        
        if not isinstance(data, dict) or not data:
            return jsonify({'error': 'Request body must be a non-empty JSON object'}), 400
        
        service = get_detection_service()
        result = service.detect_single(data)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/detect/batch', methods=['POST'])
def detect_batch():
    """Detect batch of network records."""
    if not verify_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json(silent=True)
        
        if not isinstance(data, dict) or 'records' not in data:
            return jsonify({'error': 'No records provided'}), 400
        
        records = data['records']
        
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            return jsonify({'error': 'Records must be a list of JSON objects'}), 400
        
        if len(records) == 0:
            return jsonify({'error': 'Empty records list'}), 400
        
        if len(records) > 1000:
            return jsonify({'error': 'Maximum 1000 records per batch'}), 400
        
        service = get_detection_service()
        results = service.detect_batch(records)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Batch detection failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


def application(environ, start_response):
    """WSGI entry point for Passenger."""
    return app(environ, start_response)


if __name__ == '__main__':
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'false').lower() in {'1', 'true', 'yes'},
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', '5000')),
    )
