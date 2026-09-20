FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends docker.io && rm -rf /var/lib/apt/lists/*

# The application only needs CPU inference. Resolve the complete dependency
# graph at once with the CPU-only PyTorch wheel pinned, so pip cannot select
# the multi-gigabyte CUDA/NVIDIA PyTorch distribution transitively.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.14.0+cpu" \
    -r requirements.txt

# Debian separates the client from the engine package. DevPulse uses the
# client against the mounted Docker Desktop socket for telemetry. Keeping this
# in its own layer preserves the large CPU-only inference dependency cache.
RUN apt-get update && apt-get install -y --no-install-recommends docker-cli && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
