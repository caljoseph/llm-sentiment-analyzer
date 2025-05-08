import os
import pandas as pd
from typing import List, Dict, Any, Iterator


class DataProcessor:
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load a dataset from various formats (CSV, JSON, etc.)
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            DataFrame containing the dataset
        """
        file_ext = os.path.splitext(dataset_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(dataset_path)
        elif file_ext == '.json':
            return pd.read_json(dataset_path)
        elif file_ext == '.jsonl':
            return pd.read_json(dataset_path, lines=True)
        elif file_ext == '.parquet':
            return pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def preprocess_review(self, review_text: str) -> str:
        """
        Preprocess a single review text
        
        Args:
            review_text: The review text to preprocess
            
        Returns:
            Preprocessed review text
        """
        # Basic preprocessing
        return review_text.strip()
    
    def get_batches(self, 
                    dataset: pd.DataFrame, 
                    review_column: str = 'review', 
                    rating_column: str = 'rating'
                   ) -> Iterator[List[Dict[str, Any]]]:
        """
        Split the dataset into batches for processing
        
        Args:
            dataset: DataFrame containing the dataset
            review_column: Name of the column containing review text
            rating_column: Name of the column containing actual ratings
            
        Yields:
            Batches of reviews with their metadata
        """
        total_samples = len(dataset)
        
        for i in range(0, total_samples, self.batch_size):
            batch = dataset.iloc[i:i+self.batch_size]
            batch_data = []
            
            for _, row in batch.iterrows():
                # Ensure the review text is a string
                review_text = str(row[review_column])
                
                # Preprocess the review
                processed_review = self.preprocess_review(review_text)
                
                # Create a record with review and ground truth
                record = {
                    'review_text': processed_review,
                    'actual_rating': int(row[rating_column]),
                    # Include additional metadata from the row
                    'metadata': {
                        col: row[col] for col in row.index 
                        if col not in [review_column, rating_column]
                    }
                }
                batch_data.append(record)
            
            yield batch_data