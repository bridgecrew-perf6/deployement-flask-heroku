# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:54:24 2020

@author: Nagesh

code taken from https://github.com/krishnaik06/Heroku-Demo/blob/master/app.py
"""

#%%


import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

#%%

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text = 'predicted Net hourly electrical energy is {} MW'.format(output))

    
if __name__ == "__main__":
    app.run(debug = True)








