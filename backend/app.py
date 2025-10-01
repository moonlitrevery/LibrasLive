"""
LibrasLive Backend Server
Flask + SocketIO server for real-time LIBRAS sign language recognition
"""

import os
import time
from collections import deque, Counter
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import numpy as np
from threading import Lock
import logging
import base64
import io

from infer import LibrasInference
from tts import TTSManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'libras_live_secret_2024'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global variables for inference and TTS
inference_engine = None
tts_manager = None
prediction_buffer = deque(maxlen=10)  # Buffer for temporal smoothing
last_prediction = ""
last_audio_file = None
prediction_lock = Lock()

# Smoothing parameters
STABILITY_THRESHOLD = 0.6  # 60% of predictions must be the same
MIN_PREDICTIONS = 5  # Minimum predictions before making a decision
PREDICTION_COOLDOWN = 1.0  # Seconds between predictions

last_prediction_time = 0

def initialize_models():
    """Initialize the inference engine and TTS manager"""
    global inference_engine, tts_manager
    
    try:
        # Initialize inference engine
        inference_engine = LibrasInference()
        logger.info("Inference engine initialized successfully")
        
        # Initialize TTS manager
        tts_manager = TTSManager()
        logger.info("TTS manager initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return False

def smooth_prediction(new_prediction):
    """Apply temporal smoothing to predictions"""
    global prediction_buffer, last_prediction, last_prediction_time
    
    current_time = time.time()
    
    # Add prediction to buffer
    prediction_buffer.append(new_prediction)
    
    # Check if we have enough predictions
    if len(prediction_buffer) < MIN_PREDICTIONS:
        return None
    
    # Check cooldown period
    if current_time - last_prediction_time < PREDICTION_COOLDOWN:
        return None
    
    # Count predictions in buffer
    prediction_counts = Counter(prediction_buffer)
    most_common = prediction_counts.most_common(1)[0]
    
    # Check if prediction is stable enough
    stability_ratio = most_common[1] / len(prediction_buffer)
    
    if stability_ratio >= STABILITY_THRESHOLD and most_common[0] != last_prediction:
        last_prediction = most_common[0]
        last_prediction_time = current_time
        return most_common[0]
    
    return None

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'LibrasLive Backend Running',
        'models_loaded': inference_engine is not None and tts_manager is not None
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files"""
    try:
        audio_path = os.path.join('temp_audio', filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype='audio/mpeg')
        else:
            return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return jsonify({'error': 'Audio serving failed'}), 500

@app.route('/repeat_last_audio')
def repeat_last_audio():
    """Serve the last generated audio file"""
    global last_audio_file
    
    if last_audio_file and os.path.exists(last_audio_file):
        return send_file(last_audio_file, mimetype='audio/mpeg')
    else:
        return jsonify({'error': 'No audio to repeat'}), 404

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {
        'connected': True,
        'models_ready': inference_engine is not None and tts_manager is not None
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('landmarks')
def handle_landmarks(data):
    """Handle incoming hand landmarks from client"""
    global inference_engine, last_audio_file
    
    if not inference_engine:
        emit('error', {'message': 'Models not initialized'})
        return
    
    try:
        # Extract landmarks data
        landmarks = data.get('landmarks', [])
        
        if not landmarks or len(landmarks) != 63:  # 21 points * 3 coordinates
            emit('error', {'message': 'Invalid landmarks data'})
            return
        
        # Convert to numpy array
        landmarks_array = np.array(landmarks, dtype=np.float32)
        
        # Get predictions from both models
        with prediction_lock:
            # Try alphabet recognition first
            alphabet_pred = inference_engine.predict_alphabet(landmarks_array)
            
            # Try phrase recognition (requires sequence)
            phrase_pred = inference_engine.predict_phrase(landmarks_array)
            
            # Determine which prediction to use
            # Priority: phrase > alphabet (if confidence is high enough)
            final_prediction = phrase_pred if phrase_pred and phrase_pred != "UNKNOWN" else alphabet_pred
            
            # Apply temporal smoothing
            smoothed_prediction = smooth_prediction(final_prediction)
            
            if smoothed_prediction:
                # Generate TTS
                audio_file = None
                try:
                    audio_file = tts_manager.generate_speech(smoothed_prediction)
                    last_audio_file = audio_file
                except Exception as e:
                    logger.error(f"TTS generation failed: {e}")
                
                # Send result to client
                result = {
                    'text': smoothed_prediction,
                    'timestamp': time.time(),
                    'confidence': 'high' if phrase_pred else 'medium',
                    'type': 'phrase' if phrase_pred and phrase_pred != "UNKNOWN" else 'letter'
                }
                
                if audio_file:
                    result['audio_url'] = f'/audio/{os.path.basename(audio_file)}'
                
                emit('translation', result)
                logger.info(f"Translation sent: {smoothed_prediction}")
    
    except Exception as e:
        logger.error(f"Error processing landmarks: {e}")
        emit('error', {'message': f'Processing error: {str(e)}'})

@socketio.on('request_repeat')
def handle_repeat_request():
    """Handle repeat audio request"""
    global last_audio_file
    
    if last_audio_file and os.path.exists(last_audio_file):
        emit('repeat_audio', {
            'audio_url': f'/audio/{os.path.basename(last_audio_file)}',
            'timestamp': time.time()
        })
    else:
        emit('error', {'message': 'No audio to repeat'})

@socketio.on('reset_buffer')
def handle_reset_buffer():
    """Reset the prediction buffer"""
    global prediction_buffer, last_prediction
    
    with prediction_lock:
        prediction_buffer.clear()
        last_prediction = ""
    
    emit('status', {'message': 'Buffer reset', 'timestamp': time.time()})
    logger.info("Prediction buffer reset")

@socketio.on('get_status')
def handle_status_request():
    """Send current system status"""
    status = {
        'models_loaded': inference_engine is not None and tts_manager is not None,
        'buffer_size': len(prediction_buffer),
        'last_prediction': last_prediction,
        'audio_available': last_audio_file is not None and os.path.exists(last_audio_file) if last_audio_file else False
    }
    emit('status', status)

if __name__ == '__main__':
    # Create temp directory for audio files
    os.makedirs('temp_audio', exist_ok=True)
    
    # Initialize models
    logger.info("Initializing LibrasLive backend...")
    if initialize_models():
        logger.info("All models initialized successfully")
    else:
        logger.warning("Some models failed to initialize - check model files")
    
    # Start the server
    logger.info("Starting LibrasLive backend server on http://localhost:5000")
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to True for development
        allow_unsafe_werkzeug=True
    )