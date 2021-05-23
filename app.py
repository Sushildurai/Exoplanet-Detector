# Backend Flask for heart disease detector
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect, url_for
import pickle
import sys


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.ndimage.filters import uniform_filter1d

#keras dependency 2
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam,SGD


app = Flask(__name__,template_folder='templates')
#model = pickle.load(open('exoplanet_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('home_page.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    model = pickle.load(open('exoplanet_model.pkl', 'rb'))

    #load the test dataset:
    INPUT_LIB = 'E:/8th sem/nasa kaggle dataset/'
    raw_data = np.loadtxt('exoTest_2.csv', skiprows=1, delimiter=',')
    x_test = raw_data[:, 1:]
    y_test = raw_data[:, 0, np.newaxis] - 1.

    #standardization:
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
          np.std(x_test, axis=1).reshape(-1,1))

    #moving average:
    x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)


    #final_features = [np.array(int_features)]
    predicted_model = model.predict(x_test)

    # output = round(prediction[0], 2)
    #output = prediction[0]

    
    pred=[]
    for i in range(len(predicted_model)):
        if predicted_model[i][0]<0.001:
            pred.append(0)
        elif predicted_model[i][0]>0.001:
            pred.append(1)
    

    my_array= raw_data[0,1:]
    np.set_printoptions(threshold=sys.maxsize)
    
    if pred[0] == 0:
        return render_template('home_page.html', prediction_text='Not exoplanet Detected: {}'.format(pred[0]))

    elif pred[0] == 1:
        return render_template('home_page.html', prediction_text='Exoplanet Detected: {}'.format(pred[0]),dataset_text='Dataset Used : {}'.format(my_array))


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False,threaded=False)
