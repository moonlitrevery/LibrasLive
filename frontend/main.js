/**
 * LibrasLive Frontend JavaScript
 * Handles MediaPipe Hands integration, Socket.IO communication, and UI interactions
 */

// Global variables
let socket = null;
let hands = null;
let camera = null;
let webcamElement = null;
let canvasElement = null;
let canvasCtx = null;
let isConnected = false;
let isCameraActive = false;
let showLandmarks = false;
let isProcessing = false;

// Performance tracking
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;
let predictionsCount = 0;
let lastPredictionReset = Date.now();

// Translation history
let translationHistory = [];
const maxHistoryItems = 10;

// Audio management
let currentAudio = null;
let lastAudioUrl = null;

// DOM elements
const elements = {};

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initializeSocketConnection();
    initializeEventListeners();
    updateUIStatus();
});

/**
 * Cache DOM elements for better performance
 */
function initializeElements() {
    elements.webcam = document.getElementById('webcam');
    elements.canvas = document.getElementById('landmarks-canvas');
    elements.loadingOverlay = document.getElementById('loading-overlay');
    elements.startCamera = document.getElementById('start-camera');
    elements.stopCamera = document.getElementById('stop-camera');
    elements.toggleLandmarks = document.getElementById('toggle-landmarks');
    elements.translatedText = document.getElementById('translated-text');
    elements.confidenceLevel = document.getElementById('confidence-level');
    elements.translationType = document.getElementById('translation-type');
    elements.repeatAudio = document.getElementById('repeat-audio');
    elements.volumeSlider = document.getElementById('volume-slider');
    elements.volumeValue = document.getElementById('volume-value');
    elements.translationHistory = document.getElementById('translation-history');
    elements.clearHistory = document.getElementById('clear-history');
    elements.connectionStatus = document.getElementById('connection-status');
    elements.cameraStatus = document.getElementById('camera-status');
    elements.modelStatus = document.getElementById('model-status');
    elements.latencyMetric = document.getElementById('latency-metric');
    elements.fpsMetric = document.getElementById('fps-metric');
    elements.predictionsMetric = document.getElementById('predictions-metric');
    elements.errorModal = document.getElementById('error-modal');
    elements.errorMessage = document.getElementById('error-message');
    elements.closeErrorModal = document.getElementById('close-error-modal');
    elements.dismissError = document.getElementById('dismiss-error');
    elements.ttsAudio = document.getElementById('tts-audio');

    // Cache canvas context
    canvasElement = elements.canvas;
    canvasCtx = canvasElement.getContext('2d');
    webcamElement = elements.webcam;
}

/**
 * Initialize Socket.IO connection to backend
 */
function initializeSocketConnection() {
    console.log('Connecting to LibrasLive backend...');
    
    socket = io('http://localhost:5000', {
        transports: ['websocket', 'polling'],
        timeout: 5000,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 5
    });

    // Connection events
    socket.on('connect', () => {
        console.log('Connected to LibrasLive backend');
        isConnected = true;
        updateConnectionStatus('Conectado', 'connected');
        requestBackendStatus();
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from LibrasLive backend');
        isConnected = false;
        updateConnectionStatus('Desconectado', 'disconnected');
    });

    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        updateConnectionStatus('Erro de conexão', 'error');
        showError(`Erro de conexão: ${error.message}`);
    });

    // Backend status events
    socket.on('status', (data) => {
        console.log('Backend status:', data);
        
        if (data.models_ready !== undefined) {
            const status = data.models_ready ? 'Modelos carregados' : 'Modelos não disponíveis';
            const statusClass = data.models_ready ? 'connected' : 'error';
            updateModelStatus(status, statusClass);
        }
    });

    // Translation events
    socket.on('translation', (data) => {
        console.log('Translation received:', data);
        handleTranslationResult(data);
    });

    // Error events
    socket.on('error', (data) => {
        console.error('Backend error:', data);
        showError(data.message || 'Erro do servidor');
    });

    // Audio events
    socket.on('repeat_audio', (data) => {
        console.log('Repeat audio:', data);
        if (data.audio_url) {
            playAudio(`http://localhost:5000${data.audio_url}`);
        }
    });
}

/**
 * Initialize event listeners for UI interactions
 */
function initializeEventListeners() {
    // Camera controls
    elements.startCamera.addEventListener('click', startCamera);
    elements.stopCamera.addEventListener('click', stopCamera);
    elements.toggleLandmarks.addEventListener('click', toggleLandmarks);

    // Audio controls
    elements.repeatAudio.addEventListener('click', requestRepeatAudio);
    elements.volumeSlider.addEventListener('input', updateVolume);

    // History controls
    elements.clearHistory.addEventListener('click', clearHistory);

    // Error modal
    elements.closeErrorModal.addEventListener('click', closeErrorModal);
    elements.dismissError.addEventListener('click', closeErrorModal);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);

    // Window resize handler
    window.addEventListener('resize', resizeCanvas);
}

/**
 * Initialize MediaPipe Hands
 */
async function initializeMediaPipe() {
    try {
        console.log('Initializing MediaPipe Hands...');
        elements.loadingOverlay.style.display = 'flex';

        // Initialize MediaPipe Hands
        hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        // Configure Hands
        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        // Set up result handler
        hands.onResults(onHandsResults);

        console.log('MediaPipe Hands initialized successfully');
        return true;
    } catch (error) {
        console.error('Failed to initialize MediaPipe:', error);
        showError('Falha ao inicializar MediaPipe. Verifique sua conexão com a internet.');
        return false;
    } finally {
        elements.loadingOverlay.style.display = 'none';
    }
}

/**
 * Start camera and MediaPipe processing
 */
async function startCamera() {
    try {
        if (!hands && !(await initializeMediaPipe())) {
            return;
        }

        console.log('Starting camera...');
        elements.startCamera.disabled = true;

        // Initialize camera
        camera = new Camera(webcamElement, {
            onFrame: async () => {
                if (hands && !isProcessing) {
                    isProcessing = true;
                    try {
                        await hands.send({ image: webcamElement });
                        updateFPS();
                    } finally {
                        isProcessing = false;
                    }
                }
            },
            width: 640,
            height: 480
        });

        await camera.start();

        isCameraActive = true;
        elements.stopCamera.disabled = false;
        elements.toggleLandmarks.disabled = false;
        updateCameraStatus('Câmera ativa', 'connected');
        
        // Resize canvas to match video
        resizeCanvas();

        console.log('Camera started successfully');
    } catch (error) {
        console.error('Failed to start camera:', error);
        showError('Falha ao acessar a câmera. Verifique as permissões.');
        elements.startCamera.disabled = false;
        updateCameraStatus('Erro na câmera', 'error');
    }
}

/**
 * Stop camera and MediaPipe processing
 */
function stopCamera() {
    try {
        console.log('Stopping camera...');

        if (camera) {
            camera.stop();
            camera = null;
        }

        isCameraActive = false;
        elements.startCamera.disabled = false;
        elements.stopCamera.disabled = true;
        elements.toggleLandmarks.disabled = true;
        updateCameraStatus('Câmera desconectada', 'disconnected');

        // Clear canvas
        if (canvasCtx) {
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }

        console.log('Camera stopped');
    } catch (error) {
        console.error('Error stopping camera:', error);
    }
}

/**
 * Handle MediaPipe Hands results
 */
function onHandsResults(results) {
    if (!canvasCtx) return;

    // Clear canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw landmarks if enabled
    if (showLandmarks && results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
            drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 1, radius: 3 });
        }
    }

    // Process landmarks for recognition
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && isConnected) {
        const landmarks = results.multiHandLandmarks[0];
        processLandmarks(landmarks);
    }
}

/**
 * Process landmarks and send to backend
 */
function processLandmarks(landmarks) {
    if (!socket || !isConnected) return;

    try {
        // Flatten landmarks to array of 63 values (21 points * 3 coordinates)
        const flatLandmarks = [];
        
        for (const landmark of landmarks) {
            flatLandmarks.push(landmark.x, landmark.y, landmark.z);
        }

        // Send to backend with timestamp for latency measurement
        const startTime = performance.now();
        socket.emit('landmarks', {
            landmarks: flatLandmarks,
            timestamp: startTime
        });

    } catch (error) {
        console.error('Error processing landmarks:', error);
    }
}

/**
 * Handle translation result from backend
 */
function handleTranslationResult(data) {
    const { text, confidence, type, audio_url, timestamp } = data;
    
    // Calculate latency
    const latency = performance.now() - (timestamp * 1000);
    updateLatencyMetric(Math.round(latency));

    // Update prediction counter
    predictionsCount++;
    updatePredictionsMetric();

    // Update current translation display
    elements.translatedText.textContent = text;
    elements.confidenceLevel.textContent = `Confiança: ${confidence}`;
    elements.translationType.textContent = `Tipo: ${type === 'letter' ? 'Letra' : 'Frase'}`;

    // Add to history
    addToHistory(text, confidence, type);

    // Play audio if available
    if (audio_url) {
        lastAudioUrl = `http://localhost:5000${audio_url}`;
        playAudio(lastAudioUrl);
        elements.repeatAudio.disabled = false;
    }

    // Add visual feedback
    elements.translatedText.classList.add('highlight');
    setTimeout(() => {
        elements.translatedText.classList.remove('highlight');
    }, 1000);
}

/**
 * Play audio with volume control
 */
function playAudio(audioUrl) {
    try {
        if (currentAudio) {
            currentAudio.pause();
        }

        currentAudio = new Audio(audioUrl);
        currentAudio.volume = elements.volumeSlider.value / 100;
        
        currentAudio.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    } catch (error) {
        console.error('Error setting up audio:', error);
    }
}

/**
 * Request repeat of last audio
 */
function requestRepeatAudio() {
    if (socket && isConnected) {
        socket.emit('request_repeat');
    } else if (lastAudioUrl) {
        playAudio(lastAudioUrl);
    }
}

/**
 * Update volume setting
 */
function updateVolume() {
    const volume = elements.volumeSlider.value;
    elements.volumeValue.textContent = `${volume}%`;
    
    if (currentAudio) {
        currentAudio.volume = volume / 100;
    }
}

/**
 * Add translation to history
 */
function addToHistory(text, confidence, type) {
    const historyItem = {
        text,
        confidence,
        type,
        timestamp: new Date().toLocaleTimeString()
    };

    translationHistory.unshift(historyItem);
    
    // Limit history size
    if (translationHistory.length > maxHistoryItems) {
        translationHistory = translationHistory.slice(0, maxHistoryItems);
    }

    updateHistoryDisplay();
}

/**
 * Update history display
 */
function updateHistoryDisplay() {
    elements.translationHistory.innerHTML = '';

    translationHistory.forEach(item => {
        const historyElement = document.createElement('div');
        historyElement.className = 'history-item';
        historyElement.innerHTML = `
            <span class="history-text">${item.text}</span>
            <span class="history-info">${item.type === 'letter' ? 'Letra' : 'Frase'} - ${item.timestamp}</span>
        `;
        elements.translationHistory.appendChild(historyElement);
    });
}

/**
 * Clear translation history
 */
function clearHistory() {
    translationHistory = [];
    updateHistoryDisplay();
}

/**
 * Toggle landmark display
 */
function toggleLandmarks() {
    showLandmarks = !showLandmarks;
    elements.toggleLandmarks.textContent = showLandmarks ? 'Ocultar Pontos' : 'Mostrar Pontos';
    elements.toggleLandmarks.classList.toggle('active', showLandmarks);
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboardShortcuts(event) {
    if (event.ctrlKey || event.metaKey) {
        switch (event.key.toLowerCase()) {
            case 's':
                event.preventDefault();
                if (isCameraActive) {
                    stopCamera();
                } else {
                    startCamera();
                }
                break;
            case 'l':
                event.preventDefault();
                if (isCameraActive) {
                    toggleLandmarks();
                }
                break;
            case 'r':
                event.preventDefault();
                if (!elements.repeatAudio.disabled) {
                    requestRepeatAudio();
                }
                break;
        }
    }
}

/**
 * Resize canvas to match video dimensions
 */
function resizeCanvas() {
    if (webcamElement && canvasElement) {
        const rect = webcamElement.getBoundingClientRect();
        canvasElement.width = rect.width;
        canvasElement.height = rect.height;
        canvasElement.style.width = `${rect.width}px`;
        canvasElement.style.height = `${rect.height}px`;
    }
}

/**
 * Update FPS counter
 */
function updateFPS() {
    const now = performance.now();
    frameCount++;
    
    if (now - lastFrameTime >= 1000) {
        fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
        elements.fpsMetric.textContent = `${fps} fps`;
        frameCount = 0;
        lastFrameTime = now;
    }
}

/**
 * Update predictions per minute metric
 */
function updatePredictionsMetric() {
    const now = Date.now();
    if (now - lastPredictionReset >= 60000) {
        const predictionsPerMinute = Math.round(predictionsCount * 60000 / (now - lastPredictionReset));
        elements.predictionsMetric.textContent = predictionsPerMinute.toString();
        predictionsCount = 0;
        lastPredictionReset = now;
    }
}

/**
 * Update latency metric
 */
function updateLatencyMetric(latency) {
    elements.latencyMetric.textContent = `${latency} ms`;
    
    // Color code based on latency
    const latencyClass = latency < 200 ? 'good' : latency < 500 ? 'warning' : 'poor';
    elements.latencyMetric.className = `metric-value ${latencyClass}`;
}

/**
 * Update UI status indicators
 */
function updateUIStatus() {
    updateConnectionStatus();
    updateCameraStatus();
    updateModelStatus();
}

/**
 * Update connection status
 */
function updateConnectionStatus(text = 'Conectando...', status = 'disconnected') {
    const statusElement = elements.connectionStatus;
    const indicator = statusElement.querySelector('.status-indicator');
    const textElement = statusElement.querySelector('span:last-child');
    
    indicator.className = `status-indicator ${status}`;
    textElement.textContent = text;
}

/**
 * Update camera status
 */
function updateCameraStatus(text = 'Câmera desconectada', status = 'disconnected') {
    const statusElement = elements.cameraStatus;
    const indicator = statusElement.querySelector('.status-indicator');
    const textElement = statusElement.querySelector('span:last-child');
    
    indicator.className = `status-indicator ${status}`;
    textElement.textContent = text;
}

/**
 * Update model status
 */
function updateModelStatus(text = 'Modelos carregando...', status = 'disconnected') {
    const statusElement = elements.modelStatus;
    const indicator = statusElement.querySelector('.status-indicator');
    const textElement = statusElement.querySelector('span:last-child');
    
    indicator.className = `status-indicator ${status}`;
    textElement.textContent = text;
}

/**
 * Request backend status
 */
function requestBackendStatus() {
    if (socket && isConnected) {
        socket.emit('get_status');
    }
}

/**
 * Show error modal
 */
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorModal.style.display = 'block';
}

/**
 * Close error modal
 */
function closeErrorModal() {
    elements.errorModal.style.display = 'none';
}

/**
 * Utility function to check if MediaPipe scripts are loaded
 */
function checkMediaPipeLoaded() {
    return typeof Hands !== 'undefined' && typeof Camera !== 'undefined';
}

// Performance monitoring
setInterval(() => {
    if (socket && isConnected) {
        requestBackendStatus();
    }
}, 5000);

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (camera) {
        camera.stop();
    }
    if (socket) {
        socket.disconnect();
    }
});

// Export for debugging (development only)
if (typeof window !== 'undefined') {
    window.LibrasLive = {
        socket,
        hands,
        camera,
        translationHistory,
        startCamera,
        stopCamera,
        toggleLandmarks
    };
}

console.log('LibrasLive frontend initialized');