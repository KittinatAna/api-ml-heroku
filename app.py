from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the models
volume_model = tf.keras.models.load_model('path/to/your/volume_forecast_model.h5')
price_model = tf.keras.models.load_model('path/to/your/price_forecast_model.h5')

@app.route('/predict_volume', methods=['POST'])
def predict_volume():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1, 1)
    prediction = volume_model.predict(input_data)
    return jsonify({'prediction': prediction[0][0]})

@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1, 1)
    prediction = price_model.predict(input_data)
    return jsonify({'prediction': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True)
