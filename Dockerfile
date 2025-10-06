FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml uv.lock ./
COPY vibetest ./vibetest

# Install dependencies
RUN uv sync

# Create directories for repos and evidence
RUN mkdir -p /workspace/repos /workspace/evidence

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VIBETEST_EVIDENCE_DIR=/workspace/evidence

# Default command
CMD ["uv", "run", "python", "-m", "vibetest.cli"]
