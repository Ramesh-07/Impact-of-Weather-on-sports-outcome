from flask import Flask, request, render_template

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
cricket_df = pd.read_csv('cricket_dataset_with_teams.csv')

# Features and target
X = cricket_df[['Temperature', 'Humidity', 'Wind Speed', 'Venue Name', 'Team1 Name', 'Team2 Name', 'Toss Winner', 'Toss Decision']]
y = cricket_df['Team1 Win']

# One-hot encoding categorical features
X = pd.get_dummies(X, columns=['Venue Name', 'Team1 Name', 'Team2 Name', 'Toss Winner', 'Toss Decision'])

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for Random Forest (assuming historical data isn't needed as LSTM requires sequence info)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    venue_name = request.form['venue_name']
    team1_name = request.form['team1_name']
    team2_name = request.form['team2_name']
    toss_winner = request.form['toss_winner']
    toss_decision = request.form['toss_decision']

    # Create input dataframe
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Wind Speed': [wind_speed],
        'Venue Name': [venue_name],
        'Team1 Name': [team1_name],
        'Team2 Name': [team2_name],
        'Toss Winner': [toss_winner],
        'Toss Decision': [toss_decision]
    })
    input_data = pd.get_dummies(input_data, columns=['Venue Name', 'Team1 Name', 'Team2 Name', 'Toss Winner', 'Toss Decision'])

    # Fill missing columns if needed
    for col in set(X.columns) - set(input_data.columns):
        input_data[col] = 0

    input_data = input_data[X.columns]  # Align input with training data columns
    input_data_scaled = scaler.transform(input_data)  # Scale the input data

    # Predict outcome using the trained Random Forest model
    prediction = rf_model.predict(input_data_scaled)
    outcome = 'Team1 Win' if prediction[0] == 1 else 'Team1 Lose'

    return render_template('index.html', prediction_text=f'Predicted Outcome: {outcome}')


if __name__ == "__main__":
    app.run(debug=True)
