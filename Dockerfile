########################
# Stage 1 — Builder
########################
FROM python:3.12-slim AS builder

USER root

# Install system dependencies needed to build Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH" \
    UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /knowledge-graphs

# Copy dependency list and install packages
COPY requirements.txt ./requirements.txt
RUN uv pip install --requirement requirements.txt

# Copy application code
COPY src ./src
COPY app.py ./app.py
COPY config_example.env ./config_example.env


########################
# Stage 2 — Runtime
########################
FROM python:3.12-slim

USER root

# Install only lightweight runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (small single binary, no build tools)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set environment vars
ENV PATH="/root/.local/bin:$PATH" \
    UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1

WORKDIR /knowledge-graphs

# Copy installed Python packages and source code from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /knowledge-graphs /knowledge-graphs

# Expose FastAPI port
EXPOSE 8000

# ✅ Correct FastAPI entrypoint
ENTRYPOINT ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
