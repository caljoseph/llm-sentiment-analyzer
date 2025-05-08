import asyncio
from typing import List, Dict, Any, Optional, Union

from llm_sentiment.model_manager import ModelManager
from llm_sentiment.data_processor import DataProcessor
from llm_sentiment.evaluator import Evaluator
from llm_sentiment.results_manager import ResultsManager

# Default prompts
DEFAULT_SYSTEM_PROMPT = "You are an assistant that rates Amazon product reviews on a scale of 1-5 stars based on the sentiment expressed."
DEFAULT_USER_PROMPT = "Please analyze the following Amazon product review and rate it on a scale from 1 to 5 stars, where 1 is the most negative and 5 is the most positive. Provide ONLY the numerical rating (1, 2, 3, 4, or 5) without any explanation.\n\nReview: {review}"


async def evaluate_dataset(
    dataset_path: str,
    model_name: str,
    api_base: Optional[str] = None,
    review_column: str = "review",
    rating_column: str = "rating",
    batch_size: int = 10,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 50,
    output_dir: str = "results",
    **model_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a dataset of reviews against a language model
    
    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the model to use
        api_base: API base URL for local models
        review_column: Name of the review column in the dataset
        rating_column: Name of the rating column in the dataset
        batch_size: Batch size for evaluation
        system_prompt: System prompt for the model (uses default if None)
        user_prompt_template: User prompt template for the model (uses default if None)
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        output_dir: Output directory for results
        **model_kwargs: Additional model-specific keyword arguments
        
    Returns:
        Dictionary containing results and metrics
    """
    # Set up the model manager
    model_kwargs.update({
        'temperature': temperature,
        'max_tokens': max_tokens,
    })
    
    model_manager = ModelManager(
        model_name=model_name,
        api_base=api_base,
        model_kwargs=model_kwargs
    )
    
    # Set up the evaluator
    evaluator = Evaluator(
        model_manager=model_manager,
        system_prompt=system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=user_prompt_template if user_prompt_template is not None else DEFAULT_USER_PROMPT,
    )
    
    # Run the evaluation
    results, metrics = await evaluator.evaluate_dataset(
        dataset_path=dataset_path,
        review_column=review_column,
        rating_column=rating_column,
        batch_size=batch_size
    )
    
    # Save the results
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
    user_prompt_template: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 50,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Evaluate a single review against a language model
    
    Args:
        review_text: The review text to evaluate
        model_name: Name of the model to use
        actual_rating: Optional actual rating for comparison
        api_base: API base URL for local models
        system_prompt: System prompt for the model (uses default if None)
        user_prompt_template: User prompt template for the model (uses default if None)
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        **model_kwargs: Additional model-specific keyword arguments
        
    Returns:
        Dictionary containing evaluation result
    """
    # Set up the model manager
    model_kwargs.update({
        'temperature': temperature,
        'max_tokens': max_tokens,
    })
    
    model_manager = ModelManager(
        model_name=model_name,
        api_base=api_base,
        model_kwargs=model_kwargs
    )
    
    # Set up the evaluator
    evaluator = Evaluator(
        model_manager=model_manager,
        system_prompt=system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=user_prompt_template if user_prompt_template is not None else DEFAULT_USER_PROMPT,
    )
    
    # Run the evaluation
    result = await evaluator.evaluate_single(
        review_text=review_text,
        actual_rating=actual_rating
    )
    
    return result


async def batch_evaluate(
    reviews: List[Union[str, Dict[str, Any]]],
    model_name: str,
    api_base: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 50,
    **model_kwargs
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of reviews against a language model

    Args:
        reviews: List of review texts or dictionaries with 'review_text' and 'actual_rating'
        model_name: Name of the model to use
        api_base: API base URL for local models
        system_prompt: System prompt for the model (uses default if None)
        user_prompt_template: User prompt template for the model (uses default if None)
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        **model_kwargs: Additional model-specific keyword arguments

    Returns:
        List of evaluation results
    """
    # Set up the model manager
    model_kwargs.update({
        'temperature': temperature,
        'max_tokens': max_tokens,
    })

    model_manager = ModelManager(
        model_name=model_name,
        api_base=api_base,
        model_kwargs=model_kwargs
    )

    # Set up the evaluator
    evaluator = Evaluator(
        model_manager=model_manager,
        system_prompt=system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=user_prompt_template if user_prompt_template is not None else DEFAULT_USER_PROMPT,
    )

    # Prepare the batch data
    batch_data = []
    for item in reviews:
        if isinstance(item, str):
            record = {
                'review_text': item,
                'actual_rating': None
            }
        else:
            record = {
                'review_text': item['review_text'],
                'actual_rating': item.get('actual_rating')
            }
        batch_data.append(record)

    # Run the evaluation
    results = await evaluator.evaluate_batch(batch_data)

    return results

