#imports:
import pickle
from unittest import result
import pandas as pd
import numpy as np
'''
# pediction: No
country = 'Other'
variety = 'Other'
aroma = 7.42
aftertaste = 7.33
acidity = 7.42
body = 7.25
balance = 7.33
moisture = 0.0
'''
# prediction: Yes
country = 'Colombia'
variety = 'Caturra'
aroma = 7.83
aftertaste = 7.67
acidity = 7.33
body = 7.67
balance = 7.67
moisture = 0.11

cols = ['country_of_origin', 'variety', 'aroma', 'aftertaste', 'acidity', 'body', 'balance', 'moisture']
data = [country, variety, aroma, aftertaste, acidity, body, balance, moisture]
posted = pd.DataFrame(np.array(data).reshape(1,8), columns=cols)

# load model:
loaded_model = pickle.load(open('/home/jgayoso/Dropbox/JGayoso/ML-IA_4Geeks/09.Supervised_ML/Coffee_Model/models/coffee_model.pkl', 'rb'))

# apply model to data (posted)
result = loaded_model.predict(posted)

# el resultado está en numpy array, pasar a texto:
text_result = result.tolist()[0]

print(f'{text_result}')