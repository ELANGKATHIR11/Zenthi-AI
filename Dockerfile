# Base python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Grant execute permissions to start script
RUN chmod +x start.sh

# Expose Hugging Face Space port
EXPOSE 7860

# Define entrypoint
CMD ["./start.sh"]
