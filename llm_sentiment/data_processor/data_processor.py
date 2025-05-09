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
    
    def preprocess_content(self, content: str) -> str:
        """
        Preprocess a single content item
        
        Args:
            content: The content to preprocess
            
        Returns:
            Preprocessed content
        """
        # Basic preprocessing
        return content.strip()
    
    def get_batches(self, 
                    dataset: pd.DataFrame, 
                    content_column: str = 'review', 
                    label_column: str = 'rating'
                   ) -> Iterator[List[Dict[str, Any]]]:
        """
        Split the dataset into batches for processing
        
        Args:
            dataset: DataFrame containing the dataset
            content_column: Name of the column containing content (default: 'review')
            label_column: Name of the column containing actual labels/ratings (default: 'rating')
            
        Yields:
            Batches of content with their metadata
        """
        total_samples = len(dataset)
        
        for i in range(0, total_samples, self.batch_size):
            batch = dataset.iloc[i:i+self.batch_size]
            batch_data = []
            
            for _, row in batch.iterrows():
                # Ensure the content is a string
                content = str(row[content_column])
                
                # Preprocess the content
                processed_content = self.preprocess_content(content)
                
                # Create a record with content and ground truth label
                record = {
                    'content': processed_content,
                    'label': int(row[label_column]),
                    # Include additional metadata from the row
                    'metadata': {
                        col: row[col] for col in row.index 
                        if col not in [content_column, label_column]
                    }
                }
                batch_data.append(record)
            
            yield batch_data