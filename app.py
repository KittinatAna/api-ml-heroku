import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load TFLite models
interpreter_volume = tf.lite.Interpreter(model_path='volume_forecast_model.tflite')
interpreter_price = tf.lite.Interpreter(model_path='price_forecast_model.tflite')

interpreter_volume.allocate_tensors()
interpreter_price.allocate_tensors()

input_details_volume = interpreter_volume.get_input_details()
output_details_volume = interpreter_volume.get_output_details()
input_details_price = interpreter_price.get_input_details()
output_details_price = interpreter_price.get_output_details()

def predict_volume(input_data):
    try:
        interpreter_volume.set_tensor(input_details_volume[0]['index'], input_data)
        interpreter_volume.invoke()
        output_data = interpreter_volume.get_tensor(output_details_volume[0]['index'])
        return output_data
    except Exception as e:
        logging.error(f"Error in volume prediction: {e}")
        return None

def predict_price(input_data):
    try:
        interpreter_price.set_tensor(input_details_price[0]['index'], input_data)
        interpreter_price.invoke()
        output_data = interpreter_price.get_tensor(output_details_price[0]['index'])
        return output_data
    except Exception as e:
        logging.error(f"Error in price prediction: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    try:
        volume_input = np.array(data['volume_input'], dtype=np.float32).reshape(1, -1, 1)
        price_input = np.array(data['price_input'], dtype=np.float32).reshape(1, -1, 1)

        logging.info(f'Received volume input: {volume_input}')
        logging.info(f'Received price input: {price_input}')

        volume_prediction = predict_volume(volume_input)
        price_prediction = predict_price(price_input)

        if volume_prediction is None or price_prediction is None:
            return jsonify({'error': 'Prediction error'}), 500

        volume_prediction = volume_prediction.tolist()
        price_prediction = price_prediction.tolist()

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
