FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    unzip \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install basic ML requirements
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    plotly \
    ipywidgets

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /workspace

# # Copy project files
# COPY pyproject.toml uv.lock ./
# COPY vibetest ./vibetest
# COPY README.md .

# # Install dependencies
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# RUN uv sync

# Create directories for repos and evidence
# RUN mkdir -p /workspace/repos /workspace/evidence

# COPY example_ml_repo /workspace/repos/example_ml_repo
# COPY abdallahashour7 /workspace/repos/example_ml_repo
# COPY titanic /kaggle/input/titanic

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VIBETEST_EVIDENCE_DIR=/workspace/evidence

# Default command
CMD ["uv", "run", "python", "-m", "vibetest.cli"]
