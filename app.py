from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load('model/house_price_model_rf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])
        ocean_proximity = request.form['ocean_proximity']

        # Prepare input for the model
        df = pd.DataFrame([[longitude, latitude, housing_median_age, total_rooms, 
                            total_bedrooms, population, households, median_income, ocean_proximity]],
                          columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                   'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])

        # One-Hot Encoding for 'ocean_proximity'
        proximity_categories = [
            'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 
            'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 
            'ocean_proximity_NEAR OCEAN'
        ]
        for col in proximity_categories:
            df[col] = 0  # Set all categories to 0

        if f"ocean_proximity_{ocean_proximity}" in df.columns:
            df[f"ocean_proximity_{ocean_proximity}"] = 1  # Set the selected category to 1

        df.drop('ocean_proximity', axis=1, inplace=True)

        # Predict house price
        prediction = model.predict(df)[0]

        return render_template('index.html', prediction=f"Estimated House Price: ${prediction:,.2f}")

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
