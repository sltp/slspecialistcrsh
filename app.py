from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Categorization function for predicted multiplier
def categorize_multiplier(odd):
    if 1.0 <= odd < 2.0:
        return "Low"
    elif 2.0 <= odd < 10.0:
        return "Middle"
    else:
        return "High"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the features from the incoming request
        data = request.get_json()
        features = data.get('features', [])
        
        # Ensure the features are received correctly
        if len(features) != 2:
            return jsonify({'error': 'Expected two features: multiplier and timestamp.'}), 400
        
        # Extract the multiplier and timestamp
        multiplier = features[0]
        timestamp = features[1]
        
        # Convert timestamp to numerical features (you might need more processing)
        time_obj = np.datetime64(timestamp)
        hour = time_obj.astype('datetime64[h]').astype(int) % 24
        minute = time_obj.astype('datetime64[m]').astype(int) % 60
        second = time_obj.astype('datetime64[s]').astype(int) % 60
        
        # Prepare input data for prediction
        model_input = np.array([[hour, minute, second, multiplier]])  # Adjust this if needed for your model
        
        # Scale the input
        scaled_input = scaler.transform(model_input)

        # Make the prediction using the regressor
        predicted_multiplier = model.predict(scaled_input)[0]
        
        # Categorize the predicted multiplier
        predicted_category = categorize_multiplier(predicted_multiplier)
        
        # Return the prediction and category as JSON
        return jsonify({
            'predicted_multiplier': predicted_multiplier,
            'predicted_category': predicted_category
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Home route to serve the HTML dashboard
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML dashboard

if __name__ == "__main__":
    app.run(debug=True)
