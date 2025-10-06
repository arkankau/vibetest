"""Demo script for vibetest."""

import asyncio
from pathlib import Path
from vibetest import TestCase, VibeTestAgent


async def demo():
    """Run a demo test."""
    print("=" * 80)
    print("Vibetest Demo")
    print("=" * 80)
    print("\nVibetest lets you test codebases with natural language!")

    print("\n1. Set up your API keys:")
    print("   cp .env.example .env")
    print("   # Edit .env and add ANTHROPIC_API_KEY or OPENAI_API_KEY")

    print("\n2. Run tests:")
    print("   # Use preset tests")
    print("   uv run vibetest /path/to/repo --test training_loss")
    print()
    print("   # Or use any custom test description")
    print('   uv run vibetest /path/to/repo --test "Model uses attention mechanisms"')

    print("\n3. Choose your model:")
    print("   # Claude (default)")
    print("   VIBETEST_MODEL=anthropic/claude-3-5-sonnet-20241022")
    print()
    print("   # OpenAI GPT-4")
    print("   VIBETEST_MODEL=openai/gpt-4")

    print("\nFor more information, see README.md or examples/")


def main():
    """Main entry point."""
    asyncio.run(demo())


if __name__ == "__main__":
    main()
