from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from infer import predict
from typing import List


class check(BaseModel):
    features: List[float]


app = FastAPI()

@app.post('/predict')
async def api_prediction(features: check):
    prediction = predict(features.features)
    return JSONResponse({"Prediction" : prediction})
