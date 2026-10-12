print("App.Py is running")
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model files
preprocessing = joblib.load("preprocessing.pkl")
feature_selector = joblib.load("feature_selector.pkl")
svm_model = joblib.load("diabetes_svm_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        pregnancies = float(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        diabetes_pedigree = float(
            request.form["DiabetesPedigreeFunction"]
        )
        age = float(request.form["Age"])

        # Arrange values in the same order as the training dataset
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]])

        # Replace invalid zero measurements with NaN
        invalid_indices = [1, 2, 3, 4, 5]
        for index in invalid_indices:
            if input_data[0, index] == 0:
                input_data[0, index] = np.nan

        # Apply preprocessing and feature selection
        processed_data = preprocessing.transform(input_data)
        selected_data = feature_selector.transform(processed_data)

        # Predict using the trained SVM model
        prediction = svm_model.predict(selected_data)[0]

        if prediction == 1:
            result = "Positive prediction: The model identifies a higher likelihood of diabetes."
        else:
            result = "Negative prediction: The model does not identify diabetes in this prediction."

        return render_template("index.html", result=result)

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            result="Please enter valid numeric values in all fields."
        )

    except Exception as error:
        return render_template(
            "index.html",
            result=f"An error occurred: {error}"
        )


if __name__ == "__main__":
    app.run(debug=True)
