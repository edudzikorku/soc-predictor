import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import Tuple, List # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

class DataLoader:
    def __init__(self, data_path: str, fields_to_drop: List[str], target_column: str = 'SOC'):
        self.data_path = data_path
        self.fields_to_drop = fields_to_drop
        self.target_column = target_column
        self.preprocessor = None
        self.feature_names = None
        
    def load_and_preprocess(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the data.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                Train features, test features, train labels, test labels.
        """
        df = pd.read_csv(self.data_path)
        df.drop(self.fields_to_drop, axis = 1, inplace = True)
        
        labels = df[self.target_column]
        features = df.drop(self.target_column, axis = 1)
        
        num_cols = features.select_dtypes(exclude = ['object']).columns.tolist()
        cat_cols = features.select_dtypes(include = ['object']).columns.tolist()
        
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_size, random_state = random_state)
        
        self.preprocessor = self._create_preprocessor(num_cols, cat_cols)
        
        train_features = self.preprocessor.fit_transform(x_train).toarray()  # type: ignore
        test_features = self.preprocessor.transform(x_test).toarray() # type: ignore
        
        self.feature_names = [name.split("__")[-1] for name in self.preprocessor.get_feature_names_out()] # type: ignore
        
        return train_features, test_features, y_train.values, y_test.values
    
    def _create_preprocessor(self, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
        """
        Create the preprocessing pipeline.

        Args:
            num_cols (List[str]): List of numerical feature column names.
            cat_cols (List[str]): List of categorical feature column names.
        Returns:
            ColumnTransformer: Preprocessing pipeline for numerical and categorical features.
        """
        # Set up the workflow for transforming numerical and categorical features.
        numeric_transformer = Pipeline(steps = [('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps = [('encoder', OneHotEncoder(handle_unknown = 'ignore'))])
        
        return ColumnTransformer(
            transformers = [
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ],
            remainder = 'passthrough'
        )
    
    def get_feature_names(self) -> List[str]:
        """
        Get cleaned feature names.
        This method returns the feature names after preprocessing.

        Returns:
            List[str]: List of feature names after preprocessing.
        """
        if self.feature_names is None:
            raise ValueError("Data must be loaded first")
        return self.feature_names
    
    def get_preprocessor(self) -> ColumnTransformer:
        """
        Get the fitted preprocessor.
        Returns the preprocessor after it has been fitted to the training data.
        
        Returns:
            ColumnTransformer: The fitted preprocessor.
        """
        if self.preprocessor is None:
            raise ValueError("Data must be loaded first")
        return self.preprocessor