#importing libraries
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import re
import pickle



app = Flask(__name__)
Best_model = pickle.load(open('model_tree.pkl', 'rb'))

@app.route("/")
def index():
    """Return the homepage."""
    return render_template("test-index.html")

@app.route("/directions")
def directions():
    """Return the homepage."""
    return render_template("directions.html")

@app.route('/predict')
def predict():
    return render_template("monitor.html",status="predict" )


@app.route('/resultCR',methods = ['POST'])
def resultCRNN():
    if request.method == 'POST': 
        patient_info = request.form.to_dict()
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = Best_model.predict_proba(final_features)
        output=np.round(prediction[0,1]*100,2)
        if int(output)>=65:
            prediction='Patient is at high risk'  
            risk=3      
        elif int(output)>=45:
            prediction='Patient is at moderate risk' 
            risk=2   
        else:
            prediction='Patient is healthy'
            risk=1
        print(prediction)
        return render_template('monitor.html', prediction_text='{}'.format(prediction), status="results", risk=risk)
        

if __name__ == '__main__':
    app.run()