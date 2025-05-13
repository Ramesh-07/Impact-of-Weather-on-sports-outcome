from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
cricket_df = pd.read_csv("cricket_dataset_with_teams.csv")

# Extract unique values for dropdowns
team_names = sorted(set(cricket_df["Team1 Name"]).union(set(cricket_df["Team2 Name"])))
venues = sorted(cricket_df["Venue Name"].unique())
toss_decisions = ["Bat", "Bowl"]  # Usually, these are the two possible decisions

# Features and target
X = cricket_df[['Temperature', 'Humidity', 'Wind Speed', 'Venue Name', 'Team1 Name', 'Team2 Name', 'Toss Winner', 'Toss Decision']]
y = cricket_df['Team1 Win']

# One-hot encoding categorical features
X = pd.get_dummies(X, columns=['Venue Name', 'Team1 Name', 'Team2 Name', 'Toss Winner', 'Toss Decision'])

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

app = Flask(__name__)

# Home route: Pass dropdown values to HTML
@app.route("/")
def home():
    return render_template("index1.html", teams=team_names, venues=venues, toss_decisions=toss_decisions)

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    # Print received form data for debugging
    print("Received Form Data:", request.form)

    temperature = request.form.get("temperature", type=float)
    humidity = request.form.get("humidity", type=float)
    wind_speed = request.form.get("wind_speed", type=float)
    venue_name = request.form.get("venue_name", "")
    team1_name = request.form.get("team1_name", "")
    team2_name = request.form.get("team2_name", "")
    toss_winner = request.form.get("toss_winner", "")
    toss_decision = request.form.get("toss_decision", "")

    if not all([temperature, humidity, wind_speed, venue_name, team1_name, team2_name, toss_winner, toss_decision]):
        return render_template("index1.html", prediction_text="Error: Missing input!", teams=team_names, venues=venues, toss_decisions=toss_decisions)

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind Speed": [wind_speed],
        "Venue Name": [venue_name],
        "Team1 Name": [team1_name],
        "Team2 Name": [team2_name],
        "Toss Winner": [toss_winner],
        "Toss Decision": [toss_decision]
    })

    # One-hot encode input
    input_data = pd.get_dummies(input_data, columns=["Venue Name", "Team1 Name", "Team2 Name", "Toss Winner", "Toss Decision"])

    # Fill missing columns
    for col in set(X.columns) - set(input_data.columns):
        input_data[col] = 0

    input_data = input_data[X.columns]  # Align input with training data columns
    input_data_scaled = scaler.transform(input_data)  # Scale the input data

    # Predict outcome
    prediction = rf_model.predict(input_data_scaled)

    # Determine the winner
    winning_team = team1_name if prediction[0] == 1 else team2_name
    outcome_message = f"Predicted Winner: {winning_team}"

    return render_template("index1.html", prediction_text=outcome_message, teams=team_names, venues=venues, toss_decisions=toss_decisions)

if __name__ == "__main__":
    app.run(debug=True)