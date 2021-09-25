import pickle
import sklearn

model = pickle.load(open('random_forest_classifier.pkl', 'rb'))
#prediction=model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
n = model.predict([[5,180,120,35,0,33.5,0.672,21]])
print(n)