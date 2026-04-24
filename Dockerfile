FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r src/requirements.txt

# Make infer.py executable
RUN chmod +x /app/infer.py

# Set the entrypoint
ENTRYPOINT ["python", "infer.py"]
