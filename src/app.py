from email import message
from flask import Flask, render_template, jsonify, request 
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__) # hacer ref al nombre del archivo

@app.route('/')
def hello_flask():
    return 'Hello Flask'

@app.route('/inicio')
def show_home():
    return render_template('Index.html')

'''
@app.route('/url_variables/<string: name>/<int: age>')
def url_variables(name,age):
    if age < 18:
        return jsonify(message = 'Lo siento' + name + 'no estas autorizado'), 401
    else:
        return jsonify(message = 'Bienvenido' + name), 200 # por default es 200
'''

# cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
@app.route('/<string:country>/<string:variety>/<string:variety>/<float:aroma/<float:aroma/<float:aftertaste/<float:acidity/<float:body/<float:balance/<float:moisture>')
def result(country, variety, aroma, aftertaste, acidity, body, balance, moisture) :
    cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
    data = [country, variety, aroma, aftertaste, acidity, body, balance, moisture]
    posted = pd.DataFrame(np.array(data).reshape(1,8), columns=cols)
    
    # load model:
    #loaded_model = pickle.load(open('/home/jgayoso/Dropbox/JGayoso/ML-IA_4Geeks/09.Supervised_ML/Coffee_Model/models/coffee_model.pkl', 'rb'))
    loaded_model = pickle.load(open('models/coffee_model.pkl', 'rb'))

    # apply model to data (posted)
    result = loaded_model.predict(posted)

    # el resultado está en numpy array, pasar a texto:
    text_result = result.tolist()[0]

    if text_result=='Yes' :
        return jsonify(message='Si es un café de especialidad'), 200
    else :
        return jsonify(message='No es un café de especialidad'), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 

