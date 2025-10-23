# Vibetest Quick Start

## 5-Minute Setup

### 1. Install Dependencies
After installing [uv](https://github.com/astral-sh/uv) if not already installed,
run the following to install the project and all necessary dependencies.

```bash
uv sync
```

### 2. Configure API Keys
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key(s)
# ANTHROPIC_API_KEY=your-anthropic-key-here
# OPENAI_API_KEY=your-openai-key-here
```

## Next Steps

1. Try running the examples in `examples/`
2. Understand the output and the logs written to the `logs/` directory. You will
need to install the inspect-ai plugin for vscode for the easiest way to
visualize the logs.
