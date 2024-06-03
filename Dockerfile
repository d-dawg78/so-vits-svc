FROM nvidia/cuda:12.0.0-devel-ubuntu20.04 as base

# Disable interactive shell during docker build
ENV DEBIAN_FRONTEND noninteractive

# Install python3.9 and other essentials
RUN apt-get update && \
    apt-get install -y python3-pip python3.9-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl ffmpeg
RUN pip install --upgrade poetry

# Install python packages through poetry
WORKDIR /workspace

COPY README.md .
COPY Makefile .

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install --no-root

USER root