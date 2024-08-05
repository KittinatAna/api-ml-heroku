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

def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

@app.route('/')
def home():
    return "Welcome to the Coffee Price and Volume Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        volume_input = np.array(data['volume_input'], dtype=np.float32).reshape(-1, 1)
        price_input = np.array(data['price_input'], dtype=np.float32).reshape(-1, 1)

        time_steps = len(volume_input)
        
        # Scale inputs
        scaler_volume = MinMaxScaler()
        scaler_price = MinMaxScaler()
        
        scaled_volume = scaler_volume.fit_transform(volume_input)
        scaled_price = scaler_price.fit_transform(price_input)
        
        # Create datasets
        X_volume, y_volume = create_dataset(scaled_volume, time_steps)
        X_price, y_price = create_dataset(scaled_price, time_steps)
        
        # Reshape data for LSTM
        X_volume = np.reshape(X_volume, (X_volume.shape[0], X_volume.shape[1], 1))
        X_price = np.reshape(X_price, (X_price.shape[0], X_price.shape[1], 1))
        
        # Train the LSTM model for volume
        model_volume = build_lstm_model((time_steps, 1))
        model_volume.fit(X_volume, y_volume, epochs=50, batch_size=1, verbose=1)
        
        # Train the LSTM model for price
        model_price = build_lstm_model((time_steps, 1))
        model_price.fit(X_price, y_price, epochs=50, batch_size=1, verbose=1)

        # Prepare the latest data point for prediction
        latest_volume_input = scaled_volume[-time_steps:].reshape(1, time_steps, 1)
        latest_price_input = scaled_price[-time_steps:].reshape(1, time_steps, 1)

        # Predict
        volume_prediction = model_volume.predict(latest_volume_input)
        price_prediction = model_price.predict(latest_price_input)

        # Inverse transform the predictions
        volume_prediction = scaler_volume.inverse_transform(volume_prediction).tolist()
        price_prediction = scaler_price.inverse_transform(price_prediction).tolist()

        return jsonify({
            'volume_prediction': volume_prediction[0][0],
            'price_prediction': price_prediction[0][0]
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
