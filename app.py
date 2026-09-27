import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title='House Price Prediction API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load('model.pkl')
    encoders = joblib.load('encoders.pkl')
    print("Model and encoders loaded successfully")
except Exception as e:
    print(f"Error loading model or encoders: {e}")

class HouseInput(BaseModel):
    city: str
    location: str
    property_type: str
    bedrooms: int
    baths: int
    area_marla: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.get("/options")
def get_options():
    try:
        options = {}
        for col, encoder in encoders.items():
            options[col] = list(encoder.classes_)
        return options
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_price(data: HouseInput):
    try:
        input_data = {
            "city": data.city,
            "location": data.location,
            "property_type": data.property_type,
            "bedrooms": data.bedrooms,
            "baths": data.baths,
            "area_marla": data.area_marla
        }
        
        
        for col in ['city', 'location', 'property_type']:
            encoder = encoders[col]
            val = str(input_data[col])

            if val in encoder.classes_:
                input_data[col] = encoder.transform([val])[0]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{val}' value column '{col}' ke liye valid nahi hai."
                )

        df_input = pd.DataFrame([input_data])
        
         
        prediction = model.predict(df_input)[0]
        
        return {
            "status": "success",
            "predicted_price_pkr": round(float(prediction), 2),
            "formatted_price": f"PKR {prediction:,.2f}"
        }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))