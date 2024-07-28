import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

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
    interpreter_volume.set_tensor(input_details_volume[0]['index'], input_data)
    interpreter_volume.invoke()
    output_data = interpreter_volume.get_tensor(output_details_volume[0]['index'])
    return output_data

def predict_price(input_data):
    interpreter_price.set_tensor(input_details_price[0]['index'], input_data)
    interpreter_price.invoke()
    output_data = interpreter_price.get_tensor(output_details_price[0]['index'])
    return output_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    volume_input = np.array(data['volume_input'], dtype=np.float32).reshape(1, -1, 1)
    price_input = np.array(data['price_input'], dtype=np.float32).reshape(1, -1, 1)
    
    volume_prediction = predict_volume(volume_input).tolist()
    price_prediction = predict_price(price_input).tolist()
    
    return jsonify({
        'volume_prediction': volume_prediction,
        'price_prediction': price_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)