import asyncio
import pandas as pd
from pathlib import Path
from llm_sentiment import evaluate_dataset


async def evaluate_multiple_models():
    # Configuration
    dataset_path = "data/sample_reviews.csv"
    models = [
        "gpt-3.5-turbo",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-5-sonnet-20240620"
        # The models used here depend on what API keys you have set in your environment
        # as well as what liteLLM supports
        # Any local models run with vLLM will work because they conform to OpenAI API spec
    ]

    # Create timestamp for output folder
    base_output_dir = f"results"
    Path(base_output_dir).mkdir(exist_ok=True)

    # Store results for comparison
    all_results = {}

    # Run evaluations for each model
    for model in models:
        print(f"\n{'=' * 50}")
        print(f"Evaluating model: {model}")
        print(f"{'=' * 50}")

        # Create model-specific output directory
        output_dir = f"{base_output_dir}/{model.replace('/', '_').replace('.', '_')}"

        # Run the evaluation
        results = await evaluate_dataset(
            dataset_path=dataset_path,
            model_name=model,
            output_dir=output_dir,
            batch_size=2,
        )

        # Store results
        all_results[model] = results

        # Print metrics
        print(f"\nResults for {model}:")
        for metric, value in results['metrics'].items():
            if metric in ['mean_absolute_error', 'exact_match_rate', 'within_1_accuracy']:
                print(f"  {metric}: {value:.4f}")

    # Compare models side by side
    print("\n\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Create comparison dataframe
    metrics_to_compare = ['mean_absolute_error', 'exact_match_rate', 'within_1_accuracy']
    comparison_data = {}

    for model in models:
        model_metrics = {}
        for metric in metrics_to_compare:
            model_metrics[metric] = all_results[model]['metrics'][metric]
        comparison_data[model] = model_metrics

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)

    # Save comparison results
    comparison_path = f"{base_output_dir}/model_comparison.csv"
    comparison_df.to_csv(comparison_path)
    print(f"\nComparison saved to {comparison_path}")

    # Return the combined results for further analysis if needed
    return all_results, comparison_df


if __name__ == "__main__":
    # Run the async evaluation function
    results, comparison = asyncio.run(evaluate_multiple_models())
