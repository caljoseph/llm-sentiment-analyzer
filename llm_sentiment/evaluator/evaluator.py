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
        user_prompt: str = "Please analyze the following Amazon product review and rate it on a scale from 1 to 5 stars, where 1 is the most negative and 5 is the most positive. Provide ONLY the numerical rating (1, 2, 3, 4, or 5) without any explanation.\n\nReview: {content}"
    ):
        self.model_manager = model_manager
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        
    async def evaluate_single(self, content: Optional[str] = None, content_path: Optional[str] = None, label: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a single content item

        Args:
            content: The text content to evaluate (mutually exclusive with content_path)
            content_path: Path to an image file to evaluate (mutually exclusive with content)
            label: Optional actual label/rating for comparison

        Returns:
            Dictionary with evaluation results
        """
        response = await self.model_manager.generate_sentiment_rating(
            content=content,
            content_path=content_path,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt
        )
        
        # Construct the formatted user prompt based on what was provided
        formatted_user_prompt = ""
        if content is not None:
            formatted_user_prompt = self.user_prompt.format(content=content)
        else:
            formatted_user_prompt = self.user_prompt.format(content="")

        result = {
            'system_prompt': self.system_prompt,
            'user_prompt': formatted_user_prompt,
            'raw_model_output': response.raw_output.choices[0].message.content if hasattr(response.raw_output, 'choices') else str(response.raw_output),
            'pred_label': response.extracted_label,
            'logprobs': response.logprobs,
        }

        # Include the appropriate content in the result
        if content is not None:
            result['content'] = content
        else:
            result['content_path'] = content_path
        
        if label is not None:
            result['label'] = label
            result['is_correct'] = (response.extracted_label == label)
            
        return result

    async def evaluate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of content items

        Args:
            batch: List of content records with content and labels

        Returns:
            List of evaluation results
        """
        tasks = []
        for record in batch:
            task = self.evaluate_single(
                content=record['content'],
                label=record['label']
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def evaluate_dataset(
        self,
        dataset_path: str,
        content_column: str = 'review',
        label_column: str = 'rating',
        batch_size: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Evaluate a complete dataset
        
        Args:
            dataset_path: Path to the dataset
            content_column: Name of the column containing content (default: 'review')
            label_column: Name of the column containing labels/ratings (default: 'rating')
            batch_size: Size of batches for processing
            
        Returns:
            Tuple of (evaluation results, metrics)
        """
        data_processor = DataProcessor(batch_size=batch_size)
        dataset = data_processor.load_dataset(dataset_path)
        
        all_results = []
        for batch in tqdm(data_processor.get_batches(dataset, content_column, label_column)):
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
        y_true = [r['label'] for r in results if 'label' in r]
        y_pred = [r['pred_label'] for r in results if 'label' in r]
        
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
