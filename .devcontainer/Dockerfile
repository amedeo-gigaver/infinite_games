FROM mcr.microsoft.com/devcontainers/python:3.10

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update  && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
COPY requirements_dev.txt /workspace/requirements_dev.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /workspace/requirements.txt \
    && pip install --no-cache-dir -r /workspace/requirements_dev.txt

ENV PYTHONPATH="/workspace:${PYTHONPATH}"

WORKDIR /workspace/infinite_games
