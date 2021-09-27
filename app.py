from flask import Flask, render_template, request
from flask import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('random_forest_classifier.pkl', 'rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose=float(request.form['Glucose'])
        BloodPressure=int(request.form['BloodPressure'])
        SkinThickness=int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])


        prediction=model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        output=int(prediction[0])
        if output==0:
            return render_template('index.html',prediction_text="you don't have diabetics")
        else:
            return render_template('index.html',prediction_text="sorry you have diabetics")
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
