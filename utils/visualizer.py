import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns # type: ignore
import shap # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os # type: ignore
from typing import Dict, List, Union # type: ignore
from sklearn.metrics import r2_score # type: ignore

class ModelVisualizer:
    def __init__(self, save_dir: str = './plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok = True)
        plt.style.use("ggplot")
    
    def plot_shap_summary(self, shap_values: Union[np.ndarray, List], X_test_df: pd.DataFrame, model_name: str) -> None:
        """
        Plot SHAP summary for a model.

        Args:
            shap_values: SHAP values for the model (can be array or list).
            X_test_df (pd.DataFrame): DataFrame containing test features.
            model_name (str): Name of the model for which SHAP values are plotted.
        """
        # Create summary plot (bar chart)
        plt.figure(figsize = (10, 8))
        shap.summary_plot(
            shap_values,
            X_test_df,
            plot_type = "bar",
            show = False
        )
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.savefig(f'{self.save_dir}/{model_name}_shap_summary.png', 
                   dpi = 300, bbox_inches = 'tight')
        plt.close()

        # Waterfall plot for first observation 
        plt.figure(figsize=(10, 8))
        if hasattr(shap_values, '__len__') and len(shap_values) > 0:
            # For tree models, shap_values is typically a 2D array
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                shap_explanation = shap.Explanation(
                    values = shap_values[0],
                    base_values = shap_values.mean(axis = 0).mean() if hasattr(shap_values, 'mean') else 0,
                    data = X_test_df.iloc[0].values,
                    feature_names = X_test_df.columns.tolist()
                ) 
            else:
                # For linear models
                shap_explanation = shap.Explanation(
                    values = shap_values[0] if hasattr(shap_values[0], '__len__') else shap_values,
                    base_values = 0,
                    data = X_test_df.iloc[0].values,
                    feature_names = X_test_df.columns.tolist()
                )
            
            shap.plots.waterfall(shap_explanation, show = False)
            plt.title(f'SHAP Waterfall Plot - {model_name}')
            plt.savefig(f'{self.save_dir}/{model_name}_shap_waterfall.png',
                       dpi = 300, bbox_inches = 'tight')
        plt.close()

    def plot_shap_impact(self, shap_values: Union[np.ndarray, List], X_test_df: pd.DataFrame, model_name: str) -> None:
        """
        Plot SHAP impact (beeswarm) plot showing feature impact on predictions.

        Args:
            shap_values: SHAP values for the model.
            X_test_df (pd.DataFrame): DataFrame containing test features.
            model_name (str): Name of the model for which SHAP values are plotted.
        """
        plt.figure(figsize=(10, 8))
        
        # Create beeswarm plot (impact plot)
        shap.summary_plot(
            shap_values,
            X_test_df,
            plot_type = "violin",  
            show = False
        )
        plt.title(f'SHAP Impact Plot - {model_name}')
        plt.savefig(f'{self.save_dir}/{model_name}_shap_impact.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Alternative: Beeswarm plot
        plt.figure(figsize = (10, 8))
        shap.summary_plot(
            shap_values,
            X_test_df,
            show = False
        )
        plt.title(f'SHAP Beeswarm Plot - {model_name}')
        plt.savefig(f'{self.save_dir}/{model_name}_shap_beeswarm.png', 
                   dpi = 300, bbox_inches = 'tight')
        plt.close()

    def plot_shap_dependence(self, shap_values: Union[np.ndarray, List], X_test_df: pd.DataFrame, 
                           model_name: str, feature_idx: int = 0) -> None:
        """
        Plot SHAP dependence plot for a specific feature.

        Args:
            shap_values: SHAP values for the model.
            X_test_df (pd.DataFrame): DataFrame containing test features.
            model_name (str): Name of the model.
            feature_idx (int): Index of the feature to plot dependence for.
        """
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test_df,
            show = False
        )
        feature_name = X_test_df.columns[feature_idx]
        plt.title(f'SHAP Dependence Plot - {model_name} - {feature_name}')
        plt.savefig(f'{self.save_dir}/{model_name}_shap_dependence_{feature_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_regression_predictions(self, models: Dict, X_test: np.ndarray, 
                                  y_test: np.ndarray, feature_names: List[str]) -> None:
        """
        Create regression plots for multiple models.

        Args:
            models (Dict): Dictionary of model names and their corresponding trained model instances.
            X_test (np.ndarray): Test feature set.
            y_test (np.ndarray): Test labels.
            feature_names (List[str]): List of feature names for the test set.

        Returns:
            None: Displays the regression plots for each model.
        """
        n_models = len(models)
        n_cols = 2
        n_rows = (n_models + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            sns.scatterplot(x = y_test, y = y_pred, alpha = 0.5, ax = axes[idx])
            
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', 
                         label = 'Perfect Prediction')

            sns.regplot(x = y_test, y = y_pred, scatter = False, color = 'blue',
                       ax = axes[idx], label = 'Regression Line')

            axes[idx].set_title(f'{name}\nR²: {r2:.3f}')
            axes[idx].set_xlabel('Actual Values')
            axes[idx].set_ylabel('Predicted Values')
            axes[idx].legend()

        # Remove empty subplots
        for idx in range(len(models), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/all_regression_plots.png', dpi = 300, bbox_inches = 'tight')
        plt.close()
    
    def save_regression_plots(self, models: Dict, X_test: np.ndarray, 
                             y_test: np.ndarray, format: str = 'png') -> None:
        """
        Save individual regression plots for each model.
        Args:
            models (Dict): Dictionary of model names and their corresponding trained model instances.
            X_test (np.ndarray): Test feature set.
            y_test (np.ndarray): Test labels.
            format (str): Format to save the plots, e.g., 'png', 'pdf'. Defaults to 'png'.

        Returns:
            None: Saves the regression plots for each model in the specified format.
        """
        for name, model in models.items():
            plt.figure(figsize = (10, 8))
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            sns.scatterplot(x = y_test, y = y_pred, alpha = 0.5)
            sns.regplot(x = y_test, y = y_pred, scatter = False, color = 'blue', 
                        label = 'Regression Line')

            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                    label = 'Perfect Prediction')

            plt.title(f'{name} Regression Plot\nR²: {r2:.3f}')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.legend()

            filename = os.path.join(self.save_dir, f'regression_plot_{name}.{format}')
            plt.savefig(filename, dpi=300, bbox_inches = 'tight')
            plt.close()