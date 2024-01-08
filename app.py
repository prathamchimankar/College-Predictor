from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data from the CSV file
data = pd.read_csv('college_data.csv')

# Assuming the columns contain the college names and cutoff scores
colleges = data['College'].values
cutoff_scores = data.iloc[:, 1:6].values
acceptance_labels = data['Accepted'].values

# Create a dictionary to store classifiers for each college
college_classifiers = {}

# Train a RandomForestClassifier for each college
for i, college in enumerate(colleges):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(cutoff_scores[:, i].reshape(-1, 1), acceptance_labels)
    college_classifiers[college] = clf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        score_this_year = float(request.form['score'])

        prediction_results = {}
        for college, clf in college_classifiers.items():
            score_this_year_arr = np.array(score_this_year).reshape(1, -1)
            probability_of_acceptance = clf.predict_proba(score_this_year_arr)[:, 1]
            prediction_results[college] = f"{probability_of_acceptance[0]:.2f}"

        return jsonify(prediction_results)
    else:
        return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)