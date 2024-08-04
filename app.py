from flask import Flask, request, jsonify
import numpy as np
import logging
import joblib

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load models and scalers
hw_model_volume_fit = joblib.load('hw_model_volume.pkl')
hw_model_price_fit = joblib.load('hw_model_price.pkl')
scaler_volume = joblib.load('scaler_volume.pkl')
scaler_price = joblib.load('scaler_price.pkl')

def predict_volume(input_data):
    try:
        prediction = hw_model_volume_fit.forecast(steps=1)
        return prediction
    except Exception as e:
        logging.error(f"Error in volume prediction: {e}")
        return None

def predict_price(input_data):
    try:
        prediction = hw_model_price_fit.forecast(steps=1)
        return prediction
    except Exception as e:
        logging.error(f"Error in price prediction: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    try:
        volume_input = np.array(data['volume_input'], dtype=np.float32).reshape(-1, 1)
        price_input = np.array(data['price_input'], dtype=np.float32).reshape(-1, 1)

        logging.info(f'Received volume input: {volume_input}')
        logging.info(f'Received price input: {price_input}')

        volume_prediction = predict_volume(volume_input)
        price_prediction = predict_price(price_input)

        if volume_prediction is None or price_prediction is None:
            return jsonify({'error': 'Prediction error'}), 500

        # Inverse transform the predictions
        volume_prediction = scaler_volume.inverse_transform(volume_prediction.reshape(-1, 1)).tolist()
        price_prediction = scaler_price.inverse_transform(price_prediction.reshape(-1, 1)).tolist()

        logging.info(f'Volume prediction: {volume_prediction}')
        logging.info(f'Price prediction: {price_prediction}')

        return jsonify({
            'volume_prediction': volume_prediction,
            'price_prediction': price_prediction
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
