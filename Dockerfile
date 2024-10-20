FROM ubuntu:22.04
# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
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
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install watchdog
# RUN pip install pipenv==2023.11.14

# Copy Pipenv files to the container
# COPY Pipfile Pipfile
# COPY Pipfile.lock Pipfile.lock
# ENV PIPENV_VENV_IN_PROJECT=1
# # Install dependencies using Pipenv
# RUN pipenv install --dev
# RUN pipenv install watchdog
# RUN pipenv run pip install celery redis
# Copy the rest of the Django app code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run the Django app using Pipenv
# CMD pipenv run python3.11 /app/manage.py migrate && pipenv run python3.11 /app/manage.py runserver 0.0.0.0:8000
CMD python manage.py migrate && python manage.py runserver 0.0.0.0:8000
