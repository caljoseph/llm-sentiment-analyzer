import os
import sys
import argparse
import asyncio

from llm_sentiment.model_manager import ModelManager
from llm_sentiment.data_processor import DataProcessor
from llm_sentiment.evaluator import Evaluator
from llm_sentiment.results_manager import ResultsManager


async def evaluate_dataset_command(args: argparse.Namespace) -> None:
    # Create the model manager
    model_kwargs = {}
    if args.temperature is not None:
        model_kwargs['temperature'] = args.temperature
    if args.max_tokens is not None:
        model_kwargs['max_tokens'] = args.max_tokens
    
    model_manager = ModelManager(
        model_name=args.model,
        api_base=args.api_base,
        model_kwargs=model_kwargs
    )
    
    # Create the evaluator with custom prompts only if provided
    evaluator_kwargs = {}
    if args.system_prompt is not None:
        evaluator_kwargs['system_prompt'] = args.system_prompt
    if args.user_prompt is not None:
        evaluator_kwargs['user_prompt'] = args.user_prompt
    
    evaluator = Evaluator(
        model_manager=model_manager,
        **evaluator_kwargs
    )
    
    # Run the evaluation
    results, metrics = await evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        review_column=args.review_column,
        rating_column=args.rating_column,
        batch_size=args.batch_size
    )
    
    # Save the results
    results_manager = ResultsManager(output_dir=args.output_dir)
    output_path = results_manager.save_results(
        results=results,
        metrics=metrics,
        model_name=args.model,
        dataset_name=os.path.basename(args.dataset)
    )
    
    print(f"\nEvaluation complete. Results saved to: {output_path}")
    print("\nMetrics Summary:")
    for metric, value in metrics.items():
        if metric in ['accuracy', 'mean_absolute_error', 'exact_match_rate', 'within_1_accuracy']:
            print(f"  {metric}: {value:.4f}")


async def evaluate_single_command(args: argparse.Namespace) -> None:
    # Create the model manager
    model_kwargs = {}
    if args.temperature is not None:
        model_kwargs['temperature'] = args.temperature
    if args.max_tokens is not None:
        model_kwargs['max_tokens'] = args.max_tokens
    
    model_manager = ModelManager(
        model_name=args.model,
        api_base=args.api_base,
        model_kwargs=model_kwargs
    )
    
    # Create the evaluator with custom prompts only if provided
    evaluator_kwargs = {}
    if args.system_prompt is not None:
        evaluator_kwargs['system_prompt'] = args.system_prompt
    if args.user_prompt is not None:
        evaluator_kwargs['user_prompt'] = args.user_prompt
    
    evaluator = Evaluator(
        model_manager=model_manager,
        **evaluator_kwargs
    )
    
    # Run the evaluation
    actual_rating = int(args.actual_rating) if args.actual_rating is not None else None
    result = await evaluator.evaluate_single(
        review_text=args.review_text,
        actual_rating=actual_rating
    )
    
    # Print the result
    print("\nEvaluation Result:")
    print(f"Review: {result['review_text']}")
    print(f"Predicted Rating: {result['predicted_rating']}")
    
    if actual_rating is not None:
        print(f"Actual Rating: {result['actual_rating']}")
        print(f"Is Correct: {result['is_correct']}")
    
    if args.verbose:
        print("\nSystem Prompt:")
        print(result['system_prompt'])
        
        print("\nUser Prompt:")
        print(result['user_prompt'])
        
        print("\nRaw Model Output:")
        print(result['raw_model_output'])
        
        if result['logprobs'] is not None:
            print("\nLogprobs:")
            print(result['logprobs'])


def main():
    parser = argparse.ArgumentParser(description="LLM Sentiment Analysis Evaluation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dataset evaluation command
    dataset_parser = subparsers.add_parser("dataset", help="Evaluate a dataset of reviews")
    dataset_parser.add_argument("--dataset", required=True, help="Path to the dataset file")
    dataset_parser.add_argument("--model", required=True, help="Model name to use for evaluation")
    dataset_parser.add_argument("--api-base", help="API base URL for local models")
    dataset_parser.add_argument("--review-column", default="review", help="Name of the review column in the dataset")
    dataset_parser.add_argument("--rating-column", default="rating", help="Name of the rating column in the dataset")
    dataset_parser.add_argument("--batch-size", type=int, default=10, help="Batch size for evaluation")
    dataset_parser.add_argument("--system-prompt", help="System prompt for the model")
    dataset_parser.add_argument("--user-prompt", help="User prompt template for the model")
    dataset_parser.add_argument("--temperature", type=float, help="Temperature for model generation")
    dataset_parser.add_argument("--max-tokens", type=int, help="Maximum tokens for model generation")
    dataset_parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    # Single review evaluation command
    single_parser = subparsers.add_parser("single", help="Evaluate a single review")
    single_parser.add_argument("--review-text", required=True, help="Review text to evaluate")
    single_parser.add_argument("--model", required=True, help="Model name to use for evaluation")
    single_parser.add_argument("--api-base", help="API base URL for local models")
    single_parser.add_argument("--actual-rating", type=int, choices=[1, 2, 3, 4, 5], help="Actual rating for comparison")
    single_parser.add_argument("--system-prompt", help="System prompt for the model")
    single_parser.add_argument("--user-prompt", help="User prompt template for the model")
    single_parser.add_argument("--temperature", type=float, help="Temperature for model generation")
    single_parser.add_argument("--max-tokens", type=int, help="Maximum tokens for model generation")
    single_parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "dataset":
        asyncio.run(evaluate_dataset_command(args))
    elif args.command == "single":
        asyncio.run(evaluate_single_command(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()