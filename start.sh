#!/bin/bash

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve > /dev/null 2>&1 &

# Wait for Ollama to become healthy
echo "Waiting for Ollama to boot..."
until curl -s http://localhost:11434/ > /dev/null; do
    sleep 2
done
echo "Ollama is ready."

# Preload Qwen 3.5 Hybrid SLM+VLM into VRAM
echo "Preloading qwen3.5-hybrid:4b..."
ollama run qwen3.5-hybrid:4b "Ready"

echo "Qwen 3.5 Hybrid preloaded in VRAM."

# Start FastAPI application serving frontend dashboard
echo "Starting Zenthi-AI FastAPI backend on port 7860..."
exec uvicorn api.main:app --host 0.0.0.0 --port 7860
