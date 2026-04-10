FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    ripgrep \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    openai \
    scikit-learn

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV VIBETEST_EVIDENCE_DIR=/workspace/evidence

CMD ["uv", "run", "meerkat", "--help"]
