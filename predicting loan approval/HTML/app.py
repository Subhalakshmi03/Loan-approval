from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

model = pickle.load(open(r'model.pkl','rb'))


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/submit')
def submitform():
	return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predicted():
    
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    features_names=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
    df=pd.DataFrame(features_value,columns=features_names)
    output=model.predict(df)
    print(output[0])
    if output[0] == 1:
        return render_template('result.html', prediction_text='Your loan will be approved')
    else:
        return render_template('result.html', prediction_text='Your loan may not be gets approved')

if __name__ == '__main__':
	app.run(debug=True)