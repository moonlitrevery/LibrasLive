#!/bin/bash
# LibrasLive Docker Startup Script

set -e

echo "🤟 Starting LibrasLive Application"
echo "=================================="

# Set default environment variables
export FLASK_HOST=${FLASK_HOST:-"0.0.0.0"}
export FLASK_PORT=${FLASK_PORT:-5000}
export FLASK_DEBUG=${FLASK_DEBUG:-false}
export FRONTEND_PORT=${FRONTEND_PORT:-8000}

# Print environment information
echo "Environment Configuration:"
echo "- Flask Host: $FLASK_HOST"
echo "- Flask Port: $FLASK_PORT"
echo "- Frontend Port: $FRONTEND_PORT"
echo "- Debug Mode: $FLASK_DEBUG"
echo ""

# Check if models exist, if not create dummy models
if [ ! -f "/app/backend/models/alphabet_model.pt" ] || [ ! -f "/app/backend/models/phrase_model.pt" ]; then
    echo "⚠️  Model files not found. Creating dummy models for testing..."
    cd /app/backend
    python -c "
from infer import create_dummy_models
create_dummy_models()
print('✅ Dummy models created successfully')
"
    echo ""
fi

# Function to start the backend
start_backend() {
    echo "🚀 Starting LibrasLive Backend..."
    cd /app/backend
    python app.py &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
}

# Function to start the frontend server
start_frontend() {
    echo "🌐 Starting Frontend Server..."
    cd /app/frontend
    python -m http.server $FRONTEND_PORT &
    FRONTEND_PID=$!
    echo "Frontend server started with PID: $FRONTEND_PID on port $FRONTEND_PORT"
}

# Function to handle shutdown gracefully
cleanup() {
    echo ""
    echo "🛑 Shutting down LibrasLive..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill -TERM $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill -TERM $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Clean up temporary files
    rm -rf /app/backend/temp_audio/* 2>/dev/null || true
    
    echo "✅ LibrasLive shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start services
start_backend
start_frontend

echo ""
echo "🎉 LibrasLive is ready!"
echo "=================================="
echo "Backend API: http://localhost:$FLASK_PORT"
echo "Frontend UI: http://localhost:$FRONTEND_PORT"
echo ""
echo "To test the application:"
echo "1. Open your browser to http://localhost:$FRONTEND_PORT"
echo "2. Allow camera permissions when prompted"
echo "3. Click 'Iniciar Câmera' to start recognition"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Wait for background processes
while true; do
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process died, restarting..."
        start_backend
    fi
    
    # Check if frontend is still running
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend process died, restarting..."
        start_frontend
    fi
    
    sleep 10
done