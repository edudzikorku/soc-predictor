"""
DESCRIPTION:     Model persistence class for saving and loading trained models.
                 It provides methods to save the best model with a timestamp and select 
                 the best model based on evaluation metrics, such as R² and RMSE.

AUTHOR:          Edudzi
DATE:            13/06/2025
"""

import pickle # type: ignore
import os # type: ignore
from datetime import datetime as dt # type: ignore
from typing import Dict # type: ignore
import pandas as pd # type: ignore

class ModelPersistence:
    def __init__(self, output_dir: str = './models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_best_model(self, model: object, model_name: str) -> str:
        """
        Save a single model with timestamp.
        Args:
            model (object): The trained model to be saved.
            model_name (str): Name of the model for identification.

        Returns:
            str: Path to the saved model file.
        """
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.output_dir, f'best_model_{model_name}_{timestamp}.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Model saved to {model_path}")
        return model_path
    
    def select_and_save_best_model(self, metrics_df: pd.DataFrame, best_models: Dict) -> str:
        """
        Select and save the best model based on metrics.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing model performance metrics.
            best_models (Dict): Dictionary of model names and their corresponding trained model instances.  
        Returns:    
            str: Path to the saved best model file.
        """
        sorted_models = metrics_df.sort_values(['RMSE', 'R2 Score'], ascending = [True, False])
        best_model_name = sorted_models.iloc[0]['Model']
        best_model = best_models[best_model_name]
        
        return self.save_best_model(best_model, best_model_name)