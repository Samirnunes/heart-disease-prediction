from flask import Flask, request, jsonify
import pandas as pd
from predict import predict

app = Flask(__name__)

@app.route("/api/heart-disease/predict", methods=["POST"])
def predict_heart_disease():
    data = {
        "age": float(request.form.get("age")), 
        "sex": float(request.form.get("sex")), 
        "cp": float(request.form.get("cp")), 
        "trestbps": float(request.form.get("trestbps")), 
        "chol": float(request.form.get("chol")), 
        "fbs": float(request.form.get("fbs")), 
        "restecg": float(request.form.get("restecg")), 
        "thalach": float(request.form.get("thalach")), 
        "exang": float(request.form.get("exang")), 
        "oldpeak": float(request.form.get("oldpeak")), 
        "slope": float(request.form.get("slope")), 
        "ca": float(request.form.get("ca")), 
        "thal": float(request.form.get("thal"))
    }
    prediction = predict(pd.DataFrame([data]))
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)