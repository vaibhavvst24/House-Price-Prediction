from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model/house_price_model_rf.pkl")

# Define all expected features (including one-hot encoded ocean_proximity)
expected_features = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity_<1H OCEAN", "ocean_proximity_INLAND", 
    "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY", 
    "ocean_proximity_NEAR OCEAN"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        user_input = {
            "longitude": float(request.form["longitude"]),
            "latitude": float(request.form["latitude"]),
            "housing_median_age": float(request.form["housing_median_age"]),
            "total_rooms": float(request.form["total_rooms"]),
            "total_bedrooms": float(request.form["total_bedrooms"]),
            "population": float(request.form["population"]),
            "households": float(request.form["households"]),
            "median_income": float(request.form["median_income"]),
        }
        ocean_proximity = request.form["ocean_proximity"]

        # Convert input to DataFrame
        df = pd.DataFrame([user_input])

        # Create all ocean_proximity columns and set them to 0
        for col in expected_features:
            if col.startswith("ocean_proximity_"):
                df[col] = 0

        # Set the correct ocean proximity category to 1
        ocean_col = f"ocean_proximity_{ocean_proximity}"
        if ocean_col in df.columns:
            df[ocean_col] = 1

        # Ensure correct column order
        df = df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]

        return render_template("index.html", prediction=f"Estimated House Price: ${prediction:,.2f}")

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
