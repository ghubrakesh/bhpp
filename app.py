from flask import Flask, request, jsonify, url_for, render_template
import pickle
import json
import numpy as np
import pandas as pd


app = Flask(__name__, static_folder='static')

# load the pickle model
model = pickle.load(open('model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    print("hello")
    return render_template('index.html')
    
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data) # this data will be in key value pair
    print(np.array(list(data.values())).reshape(1, -1))
    # transform the data for prediction
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    # return render_template("index.html", prediction_text="The predicted house price is {}".format(output))
    return render_template("index.html", prediction_text="The predicted house price is USD {} thousand.".format(round(output, 2)))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
    # app.run(debug=True)
