import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd

app = Flask(__name__)
##loading the model
regmodel = pickle.load(open("regmodel.pkl","rb"))
scalar = pickle.load(open("scaling.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])
def perdict_api():
    #JSON is a way to write data that both humans can read and computers can understand.
    data=request.json['data']## from json we get the data in key-value pairs
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))

    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0] * 1000)

    return jsonify(int(output[0]*1000))

if __name__ == "__main__":
    app.run(debug=True)