from typing import List, Dict, Any, Optional, Union

from llm_sentiment.model_manager import ModelManager
from llm_sentiment.data_processor import DataProcessor
from llm_sentiment.evaluator import Evaluator
from llm_sentiment.results_manager import ResultsManager


async def evaluate_dataset(
        dataset_path: str,
        model_name: str,
        api_base: Optional[str] = None,
        review_column: str = "review",
        rating_column: str = "rating",
        batch_size: int = 10,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 50,
        output_dir: str = "results",
        **model_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a dataset of reviews against a language model
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
        review_column=review_column,
        rating_column=rating_column,
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
        review_text: str,
        model_name: str,
        actual_rating: Optional[int] = None,
        api_base: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 50,
        **model_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a single review against a language model
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
        review_text=review_text,
        actual_rating=actual_rating
    )

    return result
