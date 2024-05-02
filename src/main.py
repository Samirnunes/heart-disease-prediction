from fastapi import FastAPI, Form, Depends, Response
from pydantic import BaseModel
import pandas as pd
from predict import predict

app = FastAPI()

class PredictHeartDiseaseRequest(BaseModel):
    '''
    Request model for the "/api/heart-disease/predict" endpoint.
    '''
    age: float = Form(...)
    sex: float = Form(...)
    cp: float = Form(...)
    trestbps: float = Form(...)
    chol: float = Form(...)
    fbs: float = Form(...)
    restecg: float = Form(...)
    thalach: float = Form(...)
    exang: float = Form(...)
    oldpeak: float = Form(...)
    slope: float = Form(...)
    ca: float = Form(...)
    thal: float = Form(...)
    
@app.post("/api/heart-disease/predict")
async def predict_heart_disease(request: PredictHeartDiseaseRequest = Depends()):
    rows = [
        {
        "age": request.age, 
        "sex": request.sex, 
        "cp": request.cp, 
        "trestbps": request.trestbps, 
        "chol": request.chol, 
        "fbs": request.fbs, 
        "restecg": request.restecg, 
        "thalach": request.thalach, 
        "exang": request.exang, 
        "oldpeak": request.oldpeak, 
        "slope": request.slope, 
        "ca": request.ca, 
        "thal": request.thal
        }
    ]
    return Response(predict(pd.DataFrame.from_dict(rows)))
    