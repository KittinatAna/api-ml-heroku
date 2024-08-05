from flask import Flask, request, jsonify
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/')
def home():
    return "Welcome to the Coffee Price and Volume Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        volume_input = np.array(data['volume_input'], dtype=np.float32).reshape(-1, 1)
        price_input = np.array(data['price_input'], dtype=np.float32).reshape(-1, 1)

        logging.info(f"Volume input: {volume_input}")
        logging.info(f"Price input: {price_input}")

        time_steps = len(volume_input)

        # Scale inputs
        scaler_volume = MinMaxScaler()
        scaler_price = MinMaxScaler()

        scaled_volume = scaler_volume.fit_transform(volume_input)
        scaled_price = scaler_price.fit_transform(price_input)

        # Reshape data for LSTM
        X_volume = scaled_volume.reshape(1, time_steps, 1)
        X_price = scaled_price.reshape(1, time_steps, 1)

        logging.info(f"Reshaped X_volume shape: {X_volume.shape}")
        logging.info(f"Reshaped X_price shape: {X_price.shape}")

        # Build and train the LSTM model for volume
        model_volume = build_lstm_model((time_steps, 1))
        model_volume.fit(X_volume, scaled_volume, epochs=50, batch_size=1, verbose=1)

        # Build and train the LSTM model for price
        model_price = build_lstm_model((time_steps, 1))
        model_price.fit(X_price, scaled_price, epochs=50, batch_size=1, verbose=1)

        # Predict the next month's value
        volume_prediction = model_volume.predict(X_volume)
        price_prediction = model_price.predict(X_price)

        # Inverse transform the predictions
        volume_prediction = scaler_volume.inverse_transform(volume_prediction).tolist()
        price_prediction = scaler_price.inverse_transform(price_prediction).tolist()

        logging.info(f"Volume prediction: {volume_prediction}")
        logging.info(f"Price prediction: {price_prediction}")

        return jsonify({
            'volume_prediction': volume_prediction[0][0],
            'price_prediction': price_prediction[0][0]
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
