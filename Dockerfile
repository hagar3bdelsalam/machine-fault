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

# Install PyTorch CPU FIRST (important)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install rest of requirements
RUN pip install --no-cache-dir -r src/requirements.txt

RUN test -f src/model/model_epoch_75.pkl || \
    (echo "ERROR: src/model/model_epoch_75.pkl not found" && exit 1)

# Make infer.py executable
RUN chmod +x /app/infer.py

# Set the entrypoint
ENTRYPOINT ["python", "infer.py"]
