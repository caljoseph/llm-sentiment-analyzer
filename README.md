# LLM Sentiment Analyzer

A Python library for evaluating LLM performance on sentiment analysis tasks. The library compares an LLM's predicted ratings (1-5 scale) for Amazon reviews against actual ratings.

## Features

- **Unified Abstraction Layer**: Uses LiteLLM to support cloud-based models (OpenAI, Anthropic, Google) and local models served via vLLM
- **Hardware Compatibility**: Automatically validates model compatibility with available hardware
- **Comprehensive Evaluation**: Process batches of reviews and calculate detailed metrics
- **Flexible Configuration**: Customize system and user prompts, model parameters, and more
- **Detailed Output**: For each review, includes original text, prompts, raw model output, predicted rating, and comparison to ground truth
- **Future-Ready**: Designed to be extensible for future capabilities like image inputs

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import asyncio
from llm_sentiment import evaluate_single_sync

# Simple synchronous API for single review evaluation
result = evaluate_single_sync(
    review_text="This product is amazing! I absolutely love it.",
    model_name="gpt-3.5-turbo"
)

print(f"Predicted Rating: {result['predicted_rating']}")
```

## Using with Cloud Models

```python
import os
import asyncio
from llm_sentiment import evaluate_dataset

# Set your API key in environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

async def main():
    results = await evaluate_dataset(
        dataset_path="amazon_reviews.csv",
        model_name="gpt-3.5-turbo",
        review_column="review_text",
        rating_column="star_rating"
    )
    
    print(f"Evaluation accuracy: {results['metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Using with Local Models via vLLM

```python
import asyncio
from llm_sentiment import evaluate_single

async def main():
    # Assuming vLLM server is running on localhost:8000
    result = await evaluate_single(
        review_text="This product is amazing! I absolutely love it.",
        model_name="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
        api_base="http://localhost:8000"
    )
    
    print(f"Predicted Rating: {result['predicted_rating']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Starting a vLLM Server

The library integrates with vLLM for running local models. To start a vLLM server:

```bash
# Basic server
vllm serve --model meta-llama/Llama-2-7b-chat-hf --port 8000

# With quantization for lower memory usage
vllm serve --model meta-llama/Llama-2-7b-chat-hf --port 8000 --quantization awq

# Specify local model storage
vllm serve --model meta-llama/Llama-2-7b-chat-hf --port 8000 --download-dir ./models
```

## Component Overview

- **ModelManager**: Handles model interactions, hardware compatibility, and response parsing
- **DataProcessor**: Manages data loading, batching, and preprocessing
- **Evaluator**: Runs evaluations and computes metrics
- **ResultsManager**: Formats and stores results

## CLI Usage

The library provides a command-line interface for easy usage:

```bash
# Evaluate a dataset
llm-sentiment dataset --dataset amazon_reviews.csv --model gpt-3.5-turbo

# Evaluate a single review
llm-sentiment single --review-text "This product is amazing!" --model gpt-3.5-turbo
```

## Customizing Prompts

You can customize the system and user prompts used for evaluation:

```python
from llm_sentiment import evaluate_single_sync

result = evaluate_single_sync(
    review_text="This product is amazing! I absolutely love it.",
    model_name="gpt-3.5-turbo",
    system_prompt="You are an assistant that rates Amazon product reviews.",
    user_prompt_template="Rate this review from 1-5 stars: {review}"
)
```

## Testing

Run tests with pytest:

```bash
pytest
```