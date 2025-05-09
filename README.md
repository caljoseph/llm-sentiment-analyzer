# LLM Sentiment Analyzer

## 📋 What This Tool Does

This tool helps answer questions like:
- How accurate is GPT-3.5 at predicting sentiment ratings?
- How does Claude compare to GPT-4 on sentiment analysis?
- Is my local Llama model as good as cloud-based alternatives?
- How well can models evaluate images, audio descriptions, or other non-review content?

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
# Evaluate a single content item
from llm_sentiment import evaluate_single

# Async version
import asyncio

async def run_evaluation():
    result = await evaluate_single(
        content="This product is amazing! I absolutely love it.",
        model_name="gpt-3.5-turbo"
    )
    print(f"Predicted Label: {result['pred_label']}")

asyncio.run(run_evaluation())
```

## 🖥️ Command Line Usage

Analyze a dataset:

```bash
# Analyze a dataset with default column names (review, rating)
python -m llm_sentiment.cli.main dataset \
  --dataset data/sample_reviews.csv \
  --model gpt-3.5-turbo

# Specify custom content and label columns
python -m llm_sentiment.cli.main dataset \
  --dataset data/my_custom_data.csv \
  --content-column "text" \
  --label-column "score" \
  --model gpt-3.5-turbo
```

Analyze a single content item:

```bash
# Analyze a single piece of content
python -m llm_sentiment.cli.main single \
  --content "This product is amazing!" \
  --model gpt-3.5-turbo \
  --label 5  # Optional: provide the actual rating for comparison
```

## 📊 Comparing Multiple Models

```bash
python library_usage_example.py
```

This script gives an example of how you might compare several models on the same dataset and prompts

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
            if metric in ['mean_absolute_error', 'exact_match_rate', 'within_1_accuracy']:
                print(f"  {metric}: {value:.4f}")
```

## 📝 Working with Different Content Types

### Default Product Review Use Case

By default, the library works with datasets containing 'review' and 'rating' columns:

```python
import asyncio
from llm_sentiment import evaluate_dataset

async def run_evaluation():
    results = await evaluate_dataset(
        dataset_path="data/sample_reviews.csv",  # Contains 'review' and 'rating' columns
        model_name="gpt-3.5-turbo"
    )
    print(f"Mean Absolute Error: {results['metrics']['mean_absolute_error']:.4f}")

asyncio.run(run_evaluation())
```

### Custom Dataset Column Names

For datasets with different column names:

```python
import asyncio
from llm_sentiment import evaluate_dataset

async def run_evaluation():
    results = await evaluate_dataset(
        dataset_path="data/image_descriptions.csv",  # Contains 'description' and 'score' columns
        model_name="gpt-4-vision",
        content_column="description",  # Specify your content column
        label_column="score",         # Specify your label column
        batch_size=5
    )
    print(f"Mean Absolute Error: {results['metrics']['mean_absolute_error']:.4f}")

asyncio.run(run_evaluation())
```

## 📈 Cloud and Local Model Support

### Cloud-Based Models 

```python
import os
import asyncio
from llm_sentiment import evaluate_dataset

async def run_cloud_evaluation():
    # Set your API key (can also be done via environment variables)
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Run the evaluation
    results = await evaluate_dataset(
        dataset_path="data/sample_reviews.csv",
        model_name="gpt-3.5-turbo",
        batch_size=10  # Process 10 items at a time
    )
    
    print(f"Mean Absolute Error: {results['metrics']['mean_absolute_error']:.4f}")
    print(f"Exact Match Rate: {results['metrics']['exact_match_rate']:.4f}")
    print(f"Within-1 Accuracy: {results['metrics']['within_1_accuracy']:.4f}")

asyncio.run(run_cloud_evaluation())
```

### Local Models via vLLM

If you have models running locally with vLLM:

```python
import asyncio
from llm_sentiment import evaluate_single

async def evaluate_with_local_model():
    # Connect to your local vLLM server
    result = await evaluate_single(
        content="This product is amazing!",
        model_name="hosted_vllm/meta-llama/Llama-2-7b-chat-hf",
        api_base="http://localhost:8000"
    )
    
    print(f"Predicted Label: {result['pred_label']}")

asyncio.run(evaluate_with_local_model())
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

## 🔍 Customizing Prompts

You can change how the model is asked to rate content by customizing the system and user prompts:

```python
import asyncio
from llm_sentiment import evaluate_single

async def custom_prompt_evaluation():
    # Default prompts (if not specified):
    # system_prompt = "You are an assistant that rates Amazon product reviews on a scale of 1-5 stars based on the sentiment expressed."
    # user_prompt = "Please analyze the following Amazon product review and rate it on a scale from 1 to 5 stars, where 1 is the most negative and 5 is the most positive. Provide ONLY the numerical rating (1, 2, 3, 4, or 5) without any explanation.\n\nReview: {content}"
    
    result = await evaluate_single(
        content="This product is amazing!",
        model_name="gpt-3.5-turbo",
        system_prompt="You're an expert at analyzing customer sentiment.",
        user_prompt="Rate this content from 1-5: {content}"
    )
    
    print(f"Predicted Label: {result['pred_label']}")

asyncio.run(custom_prompt_evaluation())
```

## 📁 Understanding Results

The evaluation results include these key metrics:

- `mean_absolute_error`: Average distance between predicted and actual ratings
- `exact_match_rate`: Same as accuracy, the fraction of exact matches
- `within_1_accuracy`: How often the model was within 1 star of correct (e.g., predicting 4 when actual is 5)

Each result contains:
- `content`: The content that was evaluated
- `label`: The actual label/rating (if provided)
- `pred_label`: The predicted label from the model
- `is_correct`: Whether the prediction matches the actual label

Results are saved to the `results/` directory by default with:
- Summary CSV files showing results for each evaluation
- Detailed JSON report with all metrics and predictions
- Timestamps to track experiments over time

## 💡 Component Overview

- **ModelManager**: Handles interactions with language models (via LiteLLM)
- **DataProcessor**: Loads and processes datasets with flexible column mapping
- **Evaluator**: Runs evaluations and calculates accuracy metrics
- **ResultsManager**: Saves and organizes results

## 📋 Available Models

The library supports:

- OpenAI models: `gpt-3.5-turbo`, `gpt-4`, etc.
- Anthropic models: `anthropic/claude-3-haiku-20240307`, `anthropic/claude-3-5-sonnet-20240620`, etc.
- Other providers such as Google, X, and Huggingface (provided you have a valid key to your environment)
- Local models via vLLM: Any model loaded in vLLM with OpenAI API compatibility