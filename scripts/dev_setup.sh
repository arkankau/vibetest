#!/bin/bash
# Development setup script for vibetest

set -e

echo "=================================="
echo "Vibetest Development Setup"
echo "=================================="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✓ uv found"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
uv sync

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo ""
    echo "⚠️  Docker not found. Install Docker to use sandbox features."
    echo "   Visit: https://docs.docker.com/get-docker/"
else
    echo "✓ Docker found"

    # Build Docker image
    echo ""
    echo "Building Docker image..."
    docker build -t vibetest .
    echo "✓ Docker image built"
fi

# Check for ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  ANTHROPIC_API_KEY not set."
    echo "   Set it in your environment:"
    echo "   export ANTHROPIC_API_KEY='your-key-here'"
else
    echo "✓ ANTHROPIC_API_KEY found"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs evidence vibetest_output
echo "✓ Directories created"

# Run demo
echo ""
echo "Running demo..."
uv run python main.py

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Set ANTHROPIC_API_KEY if not already set"
echo "2. Try: uv run vibetest --help"
echo "3. Check out examples/ for usage examples"
echo "4. Read docs/ARCHITECTURE.md for design details"
echo ""
