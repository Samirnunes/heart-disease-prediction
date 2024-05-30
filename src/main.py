from flask import Flask, request, jsonify, render_template
import pandas as pd
from predict import predict
import os

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
        "thal": float(request.form.get("thal")),
        "user_feedback": float(request.form.get("user_feedback"))
    }
    
    new_data = pd.DataFrame([data])
    new_data_without_feedback = new_data.drop('user_feedback', axis=1)
    csv_path = "../output/feedbacks.csv" 
    
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, index_col=0)
        print("Existing CSV successfully read.")
    else:
        df_old = pd.DataFrame(columns=new_data.columns)
        print("No existing CSV found. Creating a new DataFrame.")

    df_new = pd.concat([df_old, new_data], ignore_index=True)
    df_new.to_csv(csv_path)
    print("CSV file updated successfully.")

    prediction = predict(new_data_without_feedback)
    return jsonify({"prediction": prediction})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)