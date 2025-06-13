import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Dict # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # type: ignore
from sklearn.model_selection import cross_val_score, KFold # type: ignore

class ModelEvaluator:
    """
    A class to evaluate regression models and print performance metrics.
    This class provides methods to evaluate multiple models, calculate metrics, and print a summary of the results.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def evaluate_models(self, models: Dict, X_train: np.ndarray, 
                       X_test: np.ndarray, y_train: np.ndarray, 
                       y_test: np.ndarray, cv_folds: int = 5) -> pd.DataFrame:
        """
        Evaluate models and return metrics dataframe.

        Args:
            models (Dict): Dictionary of model names and their corresponding trained model instances.
            X_train (np.ndarray): Training feature set.
            X_test (np.ndarray): Testing feature set.
            y_train (np.ndarray): Training labels.
            y_test (np.ndarray): Testing labels.
            cv_folds (int): Number of cross-validation folds. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing model performance metrics.

        """
        results = []
        # Initialize KFold for cross-validation
        kf = KFold(n_splits = cv_folds, shuffle = True, random_state = self.random_state)
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv = kf, scoring = 'neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_rmse_std = np.sqrt(-cv_scores).std()
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2 Score': r2,
                'CV RMSE': cv_rmse,
                'CV RMSE Std': cv_rmse_std
            })
        
        metrics_df = pd.DataFrame(results)
        return metrics_df.sort_values('RMSE')
    
    def print_evaluation_summary(self, metrics_df: pd.DataFrame, best_params: Dict) -> None:
        """Print formatted evaluation summary."""
        print("\n=== Model Evaluation Summary ===")
        print("\nModel Performance Metrics:")
        print(metrics_df.to_string(index=False))

        print("\nBest Parameters for Each Model:")
        for model_name, params in best_params.items():
            print(f"\n{model_name}:")
            for param, value in params.items():
                print(f"  {param}: {value}")