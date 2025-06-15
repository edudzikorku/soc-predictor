"""
DESCRIPTION:     Model trainer class for training multiple regression models using randomized 
                 search for hyperparameter tuning. It initializes models with their parameter 
                 distributions, trains them using randomized search CV, and returns the best model. 

AUTHOR:          Edudzi
DATE:            13/06/2025
"""

from typing import Dict, Tuple # type: ignore
import numpy as np # type: ignore
import warnings # type: ignore
from sklearn.model_selection import RandomizedSearchCV, KFold # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from xgboost import XGBRegressor # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from scipy.stats import uniform, randint # type: ignore

# Ignore all forms of warnings 
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = UserWarning, module = 'xgboost')

class ModelTrainer:
    """
    A class to train multiple regression models using randomized search for hyperparameter tuning.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = self._initialize_models()
        
    def _initialize_models(self) -> Dict:
        """
        Initialize models with their parameter distributions.

        Returns:
            Dict: A dictionary containing model names as keys and their corresponding model instances and parameter distributions.
        """
        return {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state = self.random_state),
                'params': {
                    'n_estimators': randint(50, 300),
                    'max_depth': [None] + list(range(5, 31)),
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'XGBoost': {
                'model': XGBRegressor(
                    random_state = self.random_state,
                    use_label_encoder = False,
                    eval_metric = 'rmse'
                ),
                'params': {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'gamma': uniform(0, 5),
                    'reg_alpha': uniform(0, 1),
                    'reg_lambda': uniform(0, 1)
                }
            }
        }
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    cv_folds: int = 5, n_iter: int = 100) -> Tuple[Dict, Dict]:
        """
        Train models using randomized search CV.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            cv_folds (int): Number of cross-validation folds.
            n_iter (int): Number of iterations for randomized search.

        Returns:
            Tuple[Dict, Dict]: 
                A tuple containing two dictionaries:
                - best_models: Best model instances for each model type.
                - best_params: Best hyperparameters for each model type.
        """
        best_models = {}
        best_params = {}
        
        # Initialize KFold cross-validation
        kf = KFold(n_splits = cv_folds, shuffle = True, random_state = self.random_state)
        
        # Iterate over each model and perform randomized search
        for name, model_info in self.models.items():
            print(f"Training {name} with RandomizedSearchCV...")

            # Initialize RandomizedSearchCV
            random_search = RandomizedSearchCV(
                model_info['model'],
                model_info['params'],
                n_iter = n_iter,
                cv = kf,
                scoring = 'neg_mean_squared_error',
                n_jobs = -1,
                random_state = self.random_state
            )
            
            random_search.fit(X_train, y_train)
            best_models[name] = random_search.best_estimator_
            best_params[name] = random_search.best_params_
            
        return best_models, best_params