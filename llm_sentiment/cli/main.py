import os
import sys
import argparse
import asyncio

from llm_sentiment.model_manager import ModelManager, ModelDoesNotSupportVisionError
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
        content_column=args.content_column,
        label_column=args.label_column,
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
        if metric in ['mean_absolute_error', 'exact_match_rate', 'within_1_accuracy']:
            print(f"  {metric}: {value:.4f}")


async def evaluate_single_command(args: argparse.Namespace) -> None:
    try:
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

        # Check if using an image with a non-vision model
        if args.content_path is not None and not model_manager.supports_vision:
            print(f"\nError: Model '{args.model}' does not support vision inputs.")
            print("Please use a vision-capable model like 'gpt-4-vision-preview' or 'claude-3-7-sonnet-latest'.")
            sys.exit(1)

        # Run the evaluation
        label = int(args.label) if args.label is not None else None

        # Handle either content or content_path
        if args.content is not None:
            result = await evaluator.evaluate_single(
                content=args.content,
                label=label
            )
        else:
            # Verify the image file exists
            if not os.path.isfile(args.content_path):
                print(f"\nError: Image file not found: {args.content_path}")
                sys.exit(1)

            result = await evaluator.evaluate_single(
                content_path=args.content_path,
                label=label
            )
    except ModelDoesNotSupportVisionError as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: File not found - {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)
    
    # Print the result
    print("\nEvaluation Result:")
    if 'content' in result:
        print(f"Content: {result['content']}")
    else:
        print(f"Content Path: {result['content_path']}")
    print(f"Predicted Label: {result['pred_label']}")
    
    if label is not None:
        print(f"Actual Label: {result['label']}")
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
    dataset_parser = subparsers.add_parser("dataset", help="Evaluate a dataset of content items")
    dataset_parser.add_argument("--dataset", required=True, help="Path to the dataset file")
    dataset_parser.add_argument("--model", required=True, help="Model name to use for evaluation")
    dataset_parser.add_argument("--api-base", help="API base URL for local models")
    dataset_parser.add_argument("--content-column", default="review", help="Name of the column containing content to evaluate (default: 'review')")
    dataset_parser.add_argument("--label-column", default="rating", help="Name of the column containing labels/ratings (default: 'rating')")
    dataset_parser.add_argument("--batch-size", type=int, default=10, help="Batch size for evaluation")
    dataset_parser.add_argument("--system-prompt", help="System prompt for the model")
    dataset_parser.add_argument("--user-prompt", help="User prompt template for the model")
    dataset_parser.add_argument("--temperature", type=float, help="Temperature for model generation")
    dataset_parser.add_argument("--max-tokens", type=int, help="Maximum tokens for model generation")
    dataset_parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    # Single content evaluation command
    single_parser = subparsers.add_parser("single", help="Evaluate a single content item")
    content_group = single_parser.add_mutually_exclusive_group(required=True)
    content_group.add_argument("--content", help="Text content to evaluate")
    content_group.add_argument("--content-path", help="Path to an image file to evaluate")
    single_parser.add_argument("--model", required=True, help="Model name to use for evaluation")
    single_parser.add_argument("--api-base", help="API base URL for local models")
    single_parser.add_argument("--label", type=int, choices=[1, 2, 3, 4, 5], help="Actual label/rating for comparison")
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