from flask import Flask, request, jsonify
import numpy as np
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

def fit_predict_volume(input_data):
    try:
        model = ExponentialSmoothing(input_data, trend='add', seasonal='add', seasonal_periods=6)
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)
        return prediction
    except Exception as e:
        logging.error(f"Error in volume prediction: {e}")
        return None

def fit_predict_price(input_data):
    try:
        model = ExponentialSmoothing(input_data, trend='add', seasonal='add', seasonal_periods=6)
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)
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

        # Fit scalers on the input data
        scaler_volume = MinMaxScaler()
        scaler_price = MinMaxScaler()
        volume_input_scaled = scaler_volume.fit_transform(volume_input)
        price_input_scaled = scaler_price.fit_transform(price_input)

        # Fit and predict using the Holt-Winters model
        volume_prediction_scaled = fit_predict_volume(volume_input_scaled)
        price_prediction_scaled = fit_predict_price(price_input_scaled)

        if volume_prediction_scaled is None or price_prediction_scaled is None:
            return jsonify({'error': 'Prediction error'}), 500

        # Inverse transform the predictions
        volume_prediction = scaler_volume.inverse_transform(volume_prediction_scaled.reshape(-1, 1)).tolist()
        price_prediction = scaler_price.inverse_transform(price_prediction_scaled.reshape(-1, 1)).tolist()

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
