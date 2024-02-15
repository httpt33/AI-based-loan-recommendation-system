from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('userdata.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    name = request.form['Name']
    phone = request.form['phone']
    pan = request.form['pan']
    email = request.form['login-email']
    otp = request.form['otp']
    applicant_income = request.form['Applicant Income']
    education = request.form['edu']
    gender = request.form['gender']
    married = request.form['married']
    employed = request.form['employed']
    loan_amount = request.form['Loan Amount']
    loan_term = request.form['Loan Amount Term']
    credit_history = request.form['Credit History']
    property_area = request.form['Property Area']

    # Here you can perform any backend processing with the form data, such as
    # making predictions or calculations based on the user inputs.
    # For demonstration purposes, let's just print the data.
    print("Name:", name)
    print("Phone Number:", phone)
    print("PAN Number:", pan)
    print("Email:", email)
    print("OTP:", otp)
    print("Applicant Income:", applicant_income)
    print("Education:", education)
    print("Gender:", gender)
    print("Married:", married)
    print("Self Employed:", employed)
    print("Loan Amount:", loan_amount)
    print("Loan Amount Term:", loan_term)
    print("Credit History:", credit_history)
    print("Property Area:", property_area)

    # Here you can return any response to the frontend, such as a prediction or result.
    # For demonstration purposes, let's just return a success message.
    return "Prediction received successfully!"

if __name__ == '__main__':
    app.run(debug=True)
