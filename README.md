# LLM Sentiment Analyzer

A Python library for evaluating how accurately large language models (LLMs) perform sentiment analysis on Amazon product reviews. The tool compares LLM-predicted ratings (on a 1-5 scale) against the actual ratings given by reviewers.

## 📋 What This Tool Does

This tool helps answer questions like:
- How accurate is GPT-3.5 at predicting sentiment ratings?
- How does Claude compare to GPT-4 on sentiment analysis?
- Is my local Llama model as good as cloud-based alternatives?

## 🚀 Getting Started

### Installation

```bash
# Install the package in development mode
pip install -e .
```

### Setting Up API Keys

For OpenAI models (GPT-3.5, GPT-4):
```bash
export OPENAI_API_KEY=your-api-key
```

For Anthropic models (Claude):
```bash
export ANTHROPIC_API_KEY=your-api-key
```

### Quick Example

```python
# Simplest way to evaluate a single review
from llm_sentiment import evaluate_single_sync

result = evaluate_single_sync(
    review_text="This product is amazing! I absolutely love it.",
    model_name="gpt-3.5-turbo"
)

print(f"Predicted Rating: {result['predicted_rating']}")
```

## 🖥️ Command Line Usage

Analyze a dataset of reviews:

```bash
# Using the module directly
python -m llm_sentiment.cli.main dataset \
  --dataset data/sample_reviews.csv \
  --model gpt-3.5-turbo
```

Analyze a single review:

```bash
# Using the module directly
python -m llm_sentiment.cli.main single \
  --review-text "This product is amazing!" \
  --model gpt-3.5-turbo
```

## 📊 Comparing Multiple Models

Our library makes it easy to compare different models on the same dataset. The included `library_usage_example.py` script demonstrates this:

```bash
python library_usage_example.py
```

This will:
1. Run each model on the same sample dataset
2. Generate detailed reports for each model
3. Create a comparison table showing which model performed best

The code compares models like this:

```python
import asyncio
from llm_sentiment import evaluate_dataset

async def evaluate_multiple_models():
    # Define models to compare
    models = [
        "gpt-3.5-turbo",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-5-sonnet-20240620"
    ]
    
    # Store results for comparison
    all_results = {}
    
    # Run evaluations for each model
    for model in models:
        print(f"Evaluating model: {model}")
        
        # Run the evaluation
        results = await evaluate_dataset(
            dataset_path="data/sample_reviews.csv",
            model_name=model,
            batch_size=2,
        )
        
        # Store results
        all_results[model] = results
        
        # Print metrics
        print(f"Results for {model}:")
        for metric, value in results['metrics'].items():
            if metric in ['accuracy', 'mean_absolute_error', 'within_1_accuracy']:
                print(f"  {metric}: {value:.4f}")
```

## 📝 How to Use With Different Models

### Cloud-Based Models (OpenAI, Anthropic, etc.)

```python
import os
from llm_sentiment import evaluate_dataset_sync

# Set your API key (can also be done via environment variables)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Run the evaluation
results = evaluate_dataset_sync(
    dataset_path="data/sample_reviews.csv",
    model_name="gpt-3.5-turbo",
    batch_size=10  # Process 10 reviews at a time
)

print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"Mean Absolute Error: {results['metrics']['mean_absolute_error']:.4f}")
print(f"Within-1 Accuracy: {results['metrics']['within_1_accuracy']:.4f}")
```

### Local Models via vLLM

If you have models running locally with vLLM:

```python
from llm_sentiment import evaluate_single_sync

# Connect to your local vLLM server
result = evaluate_single_sync(
    review_text="This product is amazing!",
    model_name="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
    api_base="http://localhost:8000"
)

print(f"Predicted Rating: {result['predicted_rating']}")
```

### Batch Processing Reviews

Process multiple reviews at once:

```python
import asyncio
from llm_sentiment import batch_evaluate

async def process_batch():
    reviews = [
        "This is the worst product I've ever used.",
        "It's okay, but not great for the price.",
        "Absolutely love it! Would recommend to anyone!"
    ]
    
    results = await batch_evaluate(
        reviews=reviews,
        model_name="gpt-3.5-turbo"
    )
    
    for review, result in zip(reviews, results):
        print(f"Review: {review}")
        print(f"Predicted Rating: {result['predicted_rating']}")
        print()

# Run the async function
asyncio.run(process_batch())
```

## 🔧 Setting Up a Local vLLM Server

To run models locally (requires GPU):

```bash
# Basic server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000

# With quantization for lower memory usage
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8000 \
  --quantization awq
```

## 📋 Available Models

The library supports:

- OpenAI models: `gpt-3.5-turbo`, `gpt-4`, etc.
- Anthropic models: `anthropic/claude-3-haiku-20240307`, `anthropic/claude-3-5-sonnet-20240620`, etc.
- Local models via vLLM: Any model loaded in vLLM with OpenAI API compatibility

## 🔍 Customizing How Models Are Prompted

You can change how the model is asked to rate reviews by customizing the system and user prompts:

```python
from llm_sentiment import evaluate_single_sync

# Default prompts (if not specified):
# system_prompt = "You are an assistant that rates Amazon product reviews on a scale of 1-5 stars based on the sentiment expressed."
# user_prompt = "Please analyze the following Amazon product review and rate it on a scale from 1 to 5 stars, where 1 is the most negative and 5 is the most positive. Provide ONLY the numerical rating (1, 2, 3, 4, or 5) without any explanation.\n\nReview: {review}"

result = evaluate_single_sync(
    review_text="This product is amazing!",
    model_name="gpt-3.5-turbo",
    system_prompt="You're an expert at analyzing customer sentiment.",
    user_prompt="Rate this Amazon review from 1-5: {review}"
)
```

## 📁 Understanding Results

The evaluation results include these key metrics:

- `mean_absolute_error`: Average distance between predicted and actual ratings
- `exact_match_rate`: Same as accuracy, the fraction of exact matches
- `within_1_accuracy`: How often the model was within 1 star of correct (e.g., predicting 4 when actual is 5)

Results are saved to the `results/` directory by default with:
- Summary CSV files showing results for each review
- Detailed JSON report with all metrics and predictions
- Timestamps to track experiments over time

## 💡 Component Overview

- **ModelManager**: Handles interactions with language models (via LiteLLM)
- **DataProcessor**: Loads and processes review datasets
- **Evaluator**: Runs evaluations and calculates accuracy metrics
- **ResultsManager**: Saves and organizes results
