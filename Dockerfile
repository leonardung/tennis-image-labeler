# Use an official PyTorch image with CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
WORKDIR /app

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
# Install Python 3.11
RUN apt-get update && \
    apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    libpq-dev \
    build-essential \
    curl \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set up the Python environment
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the Django app code into the container
COPY . /app

# Install the Python dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run the Django app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
