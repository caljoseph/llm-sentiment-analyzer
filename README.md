# LLM Sentiment Analyzer

A Python library for evaluating LLM performance on sentiment analysis tasks. Supports both text and image inputs with comprehensive metrics and easy deployment.

## Features

- **Multi-modal Support**: Text and image sentiment analysis with vision models
- **Provider Agnostic**: Works with 100+ LLM providers through LiteLLM integration
- **Async-First Architecture**: Built for performance with async/await patterns
- **Comprehensive Metrics**: MAE, exact match rate, and within-1 accuracy
- **CLI and Library Usage**: Command-line interface and programmatic API
- **Custom Prompting**: Configurable system and user prompts
- **Batch Processing**: Efficient evaluation of large datasets

## Quick Start

### Installation

```bash
pip install -e .
# Set API key for any supported provider
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export GEMINI_API_KEY=your-key
# ... or any other LiteLLM-supported provider
```

### Evaluate Single Text

```python
from llm_sentiment import evaluate_single
import asyncio

# Works with any LiteLLM-supported model
result = await evaluate_single(
    content="This product is amazing!",
    label=5,
    model_name="gpt-3.5-turbo"  # or "claude-3-haiku", "gemini-pro", etc.
)
print(f"Predicted: {result['pred_label']}")
```

### Evaluate Dataset

```python
from llm_sentiment import evaluate_dataset

results = await evaluate_dataset(
    dataset_path="data/reviews.csv",
    model_name="mistral/mistral-large-latest",  # or any provider/model
    batch_size=10
)
print(f"Accuracy: {results['metrics']['exact_match_rate']:.3f}")
```

### Command Line

```bash
# Single evaluation - works with any provider
python -m llm_sentiment.cli.main single --content "Great product!" --model claude-3-haiku

# Dataset evaluation - test across multiple providers
python -m llm_sentiment.cli.main dataset --dataset data/reviews.csv --model gemini-pro
```

## Image Support

Works with vision models for image sentiment analysis:

```python
result = await evaluate_single(
    content_path="image.jpg",
    label=4,
    model_name="gpt-4o",
    system_prompt="Rate memes 1-5 for positivity",
    user_prompt="Rate this meme: {content}"
)
```

## Local Models

Integrate with vLLM servers for local inference:

```python
result = await evaluate_single(
    content="Review text",
    model_name="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
    api_base="http://localhost:8000"
)
```

## Architecture

Built with modular components:
- **ModelManager**: LLM integration via LiteLLM
- **DataProcessor**: Dataset loading and preprocessing  
- **Evaluator**: Metrics calculation and batch processing
- **ResultsManager**: Result persistence and formatting

## Tech Stack

- **Python 3.8+** with async/await
- **LiteLLM** for 100+ provider support (OpenAI, Anthropic, Google, Cohere, Mistral, etc.)
- **vLLM** for high-performance local model serving
- **Pandas** for flexible data processing
- **JSON/CSV** result formats