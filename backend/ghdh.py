from flask import Flask, render_template,url_for, request
import flask_sqlalchemy
from sklearn.externals import joblib
import numpy as np
app = Flask(__name__)

@app.route('/')

def home():
	return render_template('personal.html')
@app.route('/prediction', methods = ['POST'])
def prediction():
	if request.method == 'POST':
		gender = request.form['Gender']
		married = request.form['Status']
		education = request.form['education']
		employ = request.form['employ']
		annual_income = request.form['aincome']
		Loan_amount = request.form['Lamount']
		Loan_amount_term = request.form['Lamount_term']
		credit_history= request.form['credit_history']
		property = request.form['property_area']

	gender = gender.lower()
	married= married.lower()
	education = education.lower()
	employ = employ.lower()
	proper = proper.lower()
	error = 0
	if(employ=='yes'):
		employ = 1
	else:
		employ = 0
	if(gender=='male'):
		gender = 1
	else:
		gender = 0
	if (married=='married'):
		married=1
	else:
		married=0
	if (property=='rural'):
		property=0
	elif (property=='semiurban'):
		property=1
	else:
		property=2
	if (education=='graduate'):
		education=0
	else:
		education=1
	try:
	
		annual_income = int(annual_income)

		Loan_amount = int(Loan_amount)
		Loan_amount_term = int(Loan_amount_term)
		credit_history = int(credit_history)
		x_app = np.array([[gender, married, education,employ,annual_income,Loan_amount,Loan_amount_term,credit_history,proper]])
		model = joblib.load('Forest.pkl')
		ans = model.predict(x_app)
		if (ans==1):
			print("Congratulations your eligble for this Loan")
		else:
			print("We sad to inform that your request has not been accepted")
		return render_template('shit.html', prediction=ans)
	except ValueError:
		return render_template('error.html', prediction=1)
	

if __name__ == '__main__':
	app.run(debug=True)