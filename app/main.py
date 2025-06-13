import os 
import pickle
import pandas as pd # type: ignore
from fastapi import FastAPI # 
from pydantic import BaseModel 

class SOCPredictor(BaseModel):
    """
    Input data model for SOC prediction
    """
    average_elevation: int 
    average_temperature: float
    clay: int 
    land_class: str 
    mean_precipitation: float
    nitrogen: float
    phosphorus: float
    potassium: float
    sand: int 
    silt: int 
    soil_group: str
    soil_type: str
    sulfur: float
    zinc: float
    ph: float

    class Config:
        schema_extra = {
            "example": {
                "average_elevation": 1000,
                "average_temperature": 20.5,
                "clay": 30,
                "land_class": "isda",
                "mean_precipitation": 800.0,
                "nitrogen": 0.1,
                "phosphorus": 0.05,
                "potassium": 0.2,
                "sand": 40,
                "silt": 30,
                "soil_group": "isda",
                "soil_type": "isda",
                "sulfur": 0.02,
                "zinc": 0.01,
                "ph": 6.5
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title = "Soil Organic Carbon Predictor",
    description = "Predicts Soil Organic Carbon (SOC) using environmental and soil features.",
    version = "1.0.0"
)

# Load the trained model
model_path = os.path.join("models", "soc_predictor.pkl")
preprocessor_path = os.path.join("models", "preprocessor.pkl")

with open(preprocessor_path, "rb") as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)
    
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Create the prediction endpoint
@app.post("/predict")
def predict_soc(input_data: SOCPredictor):
    """
    Predict Soil Organic Carbon (SOC) based on input features.
    
    Args:
        input_data (SOCPredictor): Input data for SOC prediction.
        
    Returns:
        dict: Predicted SOC value.
    """
    # Convert input data to DataFrame
    input_df = input_data.dict()
    
    # Create a DataFrame with the same structure as the training data
    feature_order = [
        "average_elevation", "average_temperature", "clay", "land_class",
        "mean_precipitation", "nitrogen", "phosphorus", "potassium",
        "sand", "silt", "soil_group", "soil_type", "sulfur", "zinc", "ph"
    ]
    
    # Create a single-row dataframe 
    input_df = pd.DataFrame([input_df], columns = feature_order)

    # Preprocess the input data
    preprocessed_data = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    
    return {"predicted_soc": float(prediction[0]),
            "interpretation": get_iterpretation(float(prediction[0]))
            }


def get_iterpretation(soc_value: float) -> str:
    """
    Get interpretation of SOC value.
    
    Args:
        soc_value (float): Predicted SOC value.
        
    Returns:
        str: Interpretation of SOC value.
    """
    if soc_value < 1.0:
        return "Low SOC level, consider improving soil management practices."
    elif 1.0 <= soc_value < 2.0:
        return "Moderate SOC level, good for most crops."
    else:
        return "High SOC level, excellent for soil health and crop productivity."

# Add a healthcheck endpoint
@app.get("/")
def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict: Health status of the API.
    """
    return {"status": "API is running", "version": "1.0.0"}



