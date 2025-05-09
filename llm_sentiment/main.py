from typing import List, Dict, Any, Optional, Union

from llm_sentiment.model_manager import ModelManager
from llm_sentiment.data_processor import DataProcessor
from llm_sentiment.evaluator import Evaluator
from llm_sentiment.results_manager import ResultsManager


async def evaluate_dataset(
        dataset_path: str,
        model_name: str,
        api_base: Optional[str] = None,
        content_column: str = "review",
        label_column: str = "rating",
        batch_size: int = 10,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 50,
        output_dir: str = "results",
        **model_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a dataset of content items against a language model
    
    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the model to use
        api_base: Optional API base URL for the model
        content_column: Name of the column containing content to evaluate (default: 'review')
        label_column: Name of the column containing labels/ratings (default: 'rating')
        batch_size: Number of items to process in each batch
        system_prompt: Optional custom system prompt
        user_prompt: Optional custom user prompt
        temperature: Model temperature parameter
        max_tokens: Maximum tokens for model output
        output_dir: Directory to save results
        model_kwargs: Additional model parameters
        
    Returns:
        Dictionary with results, metrics, and output path
    """
    model_kwargs.update({
        'temperature': temperature,
        'max_tokens': max_tokens,
    })

    model_manager = ModelManager(
        model_name=model_name,
        api_base=api_base,
        model_kwargs=model_kwargs
    )

    evaluator_kwargs = {}
    if system_prompt is not None:
        evaluator_kwargs['system_prompt'] = system_prompt
    if user_prompt is not None:
        evaluator_kwargs['user_prompt'] = user_prompt

    evaluator = Evaluator(
        model_manager=model_manager,
        **evaluator_kwargs
    )

    results, metrics = await evaluator.evaluate_dataset(
        dataset_path=dataset_path,
        content_column=content_column,
        label_column=label_column,
        batch_size=batch_size
    )

    results_manager = ResultsManager(output_dir=output_dir)
    output_path = results_manager.save_results(
        results=results,
        metrics=metrics,
        model_name=model_name,
        dataset_name=dataset_path.split("/")[-1]
    )

    return {
        'results': results,
        'metrics': metrics,
        'output_path': output_path
    }


async def evaluate_single(
        content: str,
        model_name: str,
        label: str,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 50,
        **model_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a single content item against a language model
    
    Args:
        content: The content to evaluate
        model_name: Name of the model to use
        label: Optional actual label/rating for comparison
        api_base: Optional API base URL for the model
        system_prompt: Optional custom system prompt
        user_prompt: Optional custom user prompt
        temperature: Model temperature parameter
        max_tokens: Maximum tokens for model output
        model_kwargs: Additional model parameters
        
    Returns:
        Dictionary with evaluation results
    """
    model_kwargs.update({
        'temperature': temperature,
        'max_tokens': max_tokens,
    })

    model_manager = ModelManager(
        model_name=model_name,
        api_base=api_base,
        model_kwargs=model_kwargs
    )

    evaluator_kwargs = {}
    if system_prompt is not None:
        evaluator_kwargs['system_prompt'] = system_prompt
    if user_prompt is not None:
        evaluator_kwargs['user_prompt'] = user_prompt

    evaluator = Evaluator(
        model_manager=model_manager,
        **evaluator_kwargs
    )

    result = await evaluator.evaluate_single(
        content=content,
        label=label
    )

    return result
