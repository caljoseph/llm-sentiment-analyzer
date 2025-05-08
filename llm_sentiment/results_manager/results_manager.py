import os
import json
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional


class ResultsManager:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_results(self, 
                     results: List[Dict[str, Any]], 
                     metrics: Dict[str, float], 
                     model_name: str,
                     dataset_name: Optional[str] = None) -> str:
        """
        Save evaluation results and metrics to files
        
        Args:
            results: List of evaluation results
            metrics: Dictionary of metrics
            model_name: Name of the model used
            dataset_name: Optional name of the dataset
            
        Returns:
            Path to the saved results directory
        """
        # Create a timestamp for the results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = model_name.replace("/", "_").replace(":", "_")
        
        # Create a directory for this evaluation run
        if dataset_name:
            run_dir = os.path.join(self.output_dir, f"{model_name_safe}_{dataset_name}_{timestamp}")
        else:
            run_dir = os.path.join(self.output_dir, f"{model_name_safe}_{timestamp}")
            
        os.makedirs(run_dir, exist_ok=True)
        
        # Create a summary CSV for easy analysis
        summary_data = []
        for result in results:
            row = {
                'review_text': result.get('review_text', ''),
                'actual_rating': result.get('actual_rating', None),
                'predicted_rating': result.get('predicted_rating', None),
                'is_correct': result.get('is_correct', None)
            }
            summary_data.append(row)
            
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(run_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Clean the detailed results for JSON serialization
        clean_results = []
        for result in results:
            clean_result = {}
            for key, value in result.items():
                if key == 'raw_model_output' and not isinstance(value, (str, int, float, bool, list, dict)):
                    clean_result[key] = str(value)
                elif key == 'logprobs' and value is not None:
                    # Ensure logprobs are serializable
                    if isinstance(value, dict):
                        clean_result[key] = {}
                        for k, v in value.items():
                            if isinstance(v, (int, float)):
                                clean_result[key][str(k)] = float(v)
                            elif isinstance(v, list):
                                clean_result[key][str(k)] = [float(x) if isinstance(x, (int, float)) else str(x) for x in v]
                            else:
                                clean_result[key][str(k)] = str(v)
                    else:
                        clean_result[key] = str(value)
                else:
                    clean_result[key] = value
            clean_results.append(clean_result)
        
        # Create a consolidated run_info object
        run_info = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'timestamp': timestamp,
            'num_samples': len(results),
            # Include all metrics as a subobject
            'metrics': metrics,
            # Include detailed results as a subobject
            'detailed_results': clean_results
        }
        
        # Save the consolidated run_info
        run_info_path = os.path.join(run_dir, "run_info.json")
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
            
        return run_dir
    
    def load_results(self, results_dir: str) -> Dict[str, Any]:
        """
        Load saved evaluation results
        
        Args:
            results_dir: Path to the results directory
            
        Returns:
            Dictionary containing loaded results and metrics
        """
        # The primary data file is now run_info.json which contains everything
        run_info_path = os.path.join(results_dir, "run_info.json")
        summary_path = os.path.join(results_dir, "summary.csv")
        
        data = {}
        
        # Load run info (contains metrics and detailed results)
        if os.path.exists(run_info_path):
            with open(run_info_path, 'r') as f:
                data = json.load(f)
        
        # Load summary dataframe if available
        if os.path.exists(summary_path):
            data['summary_df'] = pd.read_csv(summary_path)
                
        return data