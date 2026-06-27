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

# Preload models into cache/memory
echo "Preloading qwen2.5:1.5b-instruct..."
ollama run qwen2.5:1.5b-instruct "Hello"

echo "Preloading qwen2.5-coder:3b..."
ollama run qwen2.5-coder:3b "Hello"

echo "Preloading riven/smolvlm:latest..."
ollama run riven/smolvlm:latest "Hello"

echo "All models preloaded."

# Start FastAPI application serving frontend dashboard
echo "Starting Zenthi-AI FastAPI backend on port 7860..."
exec uvicorn api.main:app --host 0.0.0.0 --port 7860
