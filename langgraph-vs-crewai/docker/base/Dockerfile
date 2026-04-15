# docker/base/Dockerfile
FROM python:3.11.7-slim-bookworm

# Pinned system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tiktoken pydantic pydantic-settings pyyaml

# Non-root user
RUN useradd --create-home benchuser
USER benchuser
WORKDIR /home/benchuser

# Placeholder for benchmark engine bits if needed in container
# COPY benchmark/ /home/benchuser/benchmark/
