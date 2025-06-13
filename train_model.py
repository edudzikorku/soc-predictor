import shap # type: ignore
import config # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle # type: ignore
import os # type: ignore
from utils import DataLoader, ModelTrainer, ModelEvaluator, ModelVisualizer, ModelPersistence

def main():
    # Initialize components
    data_loader = DataLoader(config.DATA_DIR, config.FIELDS_TO_DROP, config.TARGET)
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    visualizer = ModelVisualizer(config.SAVE_DIR)
    model_persistence = ModelPersistence(config.MODEL_PATH)
    
    # Load and preprocess data
    train_features, test_features, train_labels, test_labels = data_loader.load_and_preprocess()
    feature_names = data_loader.get_feature_names()
    
    # Train models
    best_models, best_params = model_trainer.train_models(train_features, train_labels)
    
    # Evaluate models
    metrics_df = model_evaluator.evaluate_models(best_models, train_features, test_features, train_labels, test_labels)
    
    # Print evaluation summary
    model_evaluator.print_evaluation_summary(metrics_df, best_params)
    
    # SHAP analysis
    X_test_df = pd.DataFrame(test_features, columns = feature_names)
    X_train_df = pd.DataFrame(train_features, columns = feature_names)
    
    # Use a sample of training data for explainer background (for performance)
    background_sample = X_train_df.sample(min(100, len(X_train_df)), random_state = config.RANDOM_STATE)
    
    for name, model in best_models.items():
        try:
            print(f"Calculating SHAP values for {name}...")
            
            if name == 'LinearRegression':
                # For linear regression, use the full training set as background
                explainer = shap.LinearExplainer(model, X_train_df)
                shap_values = explainer.shap_values(X_test_df)
            else:
                # For tree-based models, use TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_df)
            
            # Validate SHAP values
            if shap_values is not None and len(shap_values) > 0:
                print(f"SHAP values calculated successfully for {name}")
                print(f"SHAP values shape: {np.array(shap_values).shape}")
                
                # Plot SHAP summary (feature importance)
                visualizer.plot_shap_summary(shap_values, X_test_df, name)
                
                # Plot SHAP impact plot
                visualizer.plot_shap_impact(shap_values, X_test_df, name)
                
                # Plot dependence plot for the most important feature
                if len(X_test_df.columns) > 0:
                    visualizer.plot_shap_dependence(shap_values, X_test_df, name, feature_idx = 0)
                
                print(f"All SHAP plots saved for {name}")
            else:
                print(f"SHAP values are empty for {name}")
                
        except Exception as e:
            print(f"SHAP calculation failed for {name}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            # Continue with other models even if one fails
    
    # Visualization
    print("Creating regression plots...")
    visualizer.plot_regression_predictions(best_models, test_features, test_labels, feature_names)
    visualizer.save_regression_plots(best_models, test_features, test_labels)
    
    # Save best model
    model_persistence.select_and_save_best_model(metrics_df, best_models)

    # Save the preprocessor 
    preprocessor = data_loader.get_preprocessor()
    with open(os.path.join(config.MODEL_PATH, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    print("Analysis complete!")

if __name__ == "__main__":
    main()