from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open(('random_forest_classifier.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
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
        BMI = int(request.form['BMI'])
        DiabetesPedigreeFunction = int(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])


        prediction=model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        output=round(prediction[0])
        if output==0:
            return render_template('index.html',prediction_texts="you don't  have diabetics")
        else:
            return render_template('index.html',prediction_text='sorry you have diabetics')
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
