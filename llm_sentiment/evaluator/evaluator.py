import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from llm_sentiment.model_manager import ModelManager
from llm_sentiment.data_processor import DataProcessor


class Evaluator:
    def __init__(
        self,
        model_manager: ModelManager,
        system_prompt: str = "You are an assistant that rates Amazon product reviews on a scale of 1-5 stars based on the sentiment expressed.",
        user_prompt: str = "Please analyze the following Amazon product review and rate it on a scale from 1 to 5 stars, where 1 is the most negative and 5 is the most positive. Provide ONLY the numerical rating (1, 2, 3, 4, or 5) without any explanation.\n\nReview: {review}"
    ):
        self.model_manager = model_manager
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        
    async def evaluate_single(self, review_text: str, actual_rating: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a single review text
        
        Args:
            review_text: The review text to evaluate
            actual_rating: Optional actual rating for comparison
            
        Returns:
            Dictionary with evaluation results
        """
        response = await self.model_manager.generate_sentiment_rating(
            review_text=review_text,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt
        )
        
        result = {
            'review_text': review_text,
            'system_prompt': self.system_prompt,
            'user_prompt': self.user_prompt.format(review=review_text),
            'raw_model_output': response.raw_output.choices[0].message.content if hasattr(response.raw_output, 'choices') else str(response.raw_output),
            'predicted_rating': response.extracted_rating,
            'logprobs': response.logprobs,
        }
        
        if actual_rating is not None:
            result['actual_rating'] = actual_rating
            result['is_correct'] = (response.extracted_rating == actual_rating)
            
        return result

    async def evaluate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of reviews

        Args:
            batch: List of review records with text and actual ratings

        Returns:
            List of evaluation results
        """
        tasks = []
        for record in batch:
            task = self.evaluate_single(
                review_text=record['review_text'],
                actual_rating=record['actual_rating']
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def evaluate_dataset(
        self,
        dataset_path: str,
        review_column: str = 'review',
        rating_column: str = 'rating',
        batch_size: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Evaluate a complete dataset
        
        Args:
            dataset_path: Path to the dataset
            review_column: Name of the column containing review text
            rating_column: Name of the column containing actual ratings
            batch_size: Size of batches for processing
            
        Returns:
            Tuple of (evaluation results, metrics)
        """
        data_processor = DataProcessor(batch_size=batch_size)
        dataset = data_processor.load_dataset(dataset_path)
        
        all_results = []
        for batch in tqdm(data_processor.get_batches(dataset, review_column, rating_column)):
            batch_results = await self.evaluate_batch(batch)
            all_results.extend(batch_results)
            
        # Calculate metrics
        metrics = self.calculate_metrics(all_results)
        
        return all_results, metrics
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate evaluation metrics from results
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of metric names and values
        """
        y_true = [r['actual_rating'] for r in results if 'actual_rating' in r]
        y_pred = [r['predicted_rating'] for r in results if 'actual_rating' in r]
        
        if not y_true or not y_pred:
            return {}
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        
        # Calculate exact match rate (accuracy)
        exact_match = accuracy
        
        # Calculate within-1 accuracy
        within_1 = np.mean([abs(t - p) <= 1 for t, p in zip(y_true, y_pred)])
        
        metrics = {
            'mean_absolute_error': mae,
            'exact_match_rate': exact_match,
            'within_1_accuracy': within_1,
        }
            
        return metrics
