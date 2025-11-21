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
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see prediction details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'libras_live_secret_2024'
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# Disable engineio and socketio logging
logging.getLogger('engineio').setLevel(logging.WARNING)
logging.getLogger('socketio').setLevel(logging.WARNING)

# Global variables for inference and TTS
inference_engine = None
tts_manager = None
prediction_buffer = deque(maxlen=10)  # Buffer for temporal smoothing
last_prediction = ""
last_audio_file = None
prediction_lock = Lock()

# Smoothing parameters
STABILITY_THRESHOLD = 0.4  # 40% of predictions must be the same (reduced due to model uncertainty)
MIN_PREDICTIONS = 3  # Minimum predictions before making a decision (reduced for easier testing)
PREDICTION_COOLDOWN = 0.5  # Seconds between predictions (reduced for faster response)

last_prediction_time = 0
recent_predictions = deque(maxlen=20)  # Track last 20 accepted predictions to detect repetition

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
    global prediction_buffer, last_prediction, last_prediction_time, recent_predictions
    
    current_time = time.time()
    
    # Only add valid predictions to buffer (skip None)
    # This prevents None from diluting the buffer
    if new_prediction is not None:
        prediction_buffer.append(new_prediction)
    
    # Check if we have enough predictions
    if len(prediction_buffer) < MIN_PREDICTIONS:
        return None
    
    # Check cooldown period
    if current_time - last_prediction_time < PREDICTION_COOLDOWN:
        return None
    
    # Count predictions in buffer
    prediction_counts = Counter(prediction_buffer)
    if not prediction_counts:
        return None  # Buffer is empty (only None predictions)
    
    most_common = prediction_counts.most_common(1)[0]
    
    # Check if prediction is stable enough
    stability_ratio = most_common[1] / len(prediction_buffer)
    
    logger.debug(f"Buffer status: {len(prediction_buffer)} valid predictions, most common: {most_common[0]} ({most_common[1]}x), stability: {stability_ratio:.2%}")
    
    if stability_ratio >= STABILITY_THRESHOLD and most_common[0] != last_prediction:
        # Check for excessive repetition in recent predictions
        # Only block if the same letter appears in >70% of recent predictions (less aggressive)
        if len(recent_predictions) >= 15:
            recent_counts = Counter(recent_predictions)
            recent_most_common = recent_counts.most_common(1)[0]
            recent_ratio = recent_most_common[1] / len(recent_predictions)
            
            # If the same prediction appears in >70% of recent predictions, require slightly higher stability
            if recent_most_common[0] == most_common[0] and recent_ratio > 0.70:
                logger.debug(f"Excessive repetition detected: {most_common[0]} appears in {recent_ratio*100:.1f}% of recent predictions. Requiring higher stability.")
                # Require 60% stability instead of 50% to accept this prediction (less aggressive)
                if stability_ratio < 0.60:
                    logger.debug(f"Rejecting due to repetition - stability {stability_ratio:.2f} < 0.60 required")
                    return None
        
        # Accept prediction
        last_prediction = most_common[0]
        last_prediction_time = current_time
        recent_predictions.append(most_common[0])  # Track accepted prediction
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
            if alphabet_pred:
                logger.debug(f"Alphabet prediction: {alphabet_pred}")
            
            # Try phrase recognition (requires sequence)
            phrase_pred = inference_engine.predict_phrase(landmarks_array)
            if phrase_pred and phrase_pred != "UNKNOWN":
                logger.debug(f"Phrase prediction: {phrase_pred}")
            
            # Determine which prediction to use
            # Priority: phrase > alphabet (if confidence is high enough)
            final_prediction = phrase_pred if phrase_pred and phrase_pred != "UNKNOWN" else alphabet_pred
            
            if final_prediction:
                logger.debug(f"Final prediction (before smoothing): {final_prediction}, buffer size: {len(prediction_buffer)}")
            else:
                logger.debug(f"No prediction (alphabet: {alphabet_pred}, phrase: {phrase_pred}), buffer size: {len(prediction_buffer)}")
            
            # Apply temporal smoothing
            smoothed_prediction = smooth_prediction(final_prediction)
            
            if smoothed_prediction:
                logger.info(f"Smoothed prediction ready: {smoothed_prediction}")
            elif final_prediction:
                logger.debug(f"Prediction not stable yet: {final_prediction} (need {MIN_PREDICTIONS} predictions with {STABILITY_THRESHOLD*100}% stability)")
            
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