from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'fetch.php'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    pan = db.Column(db.String(20))
    email = db.Column(db.String(100))
    otp = db.Column(db.String(10))


class LoanRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    monthly_income = db.Column(db.Float)
    cibil_score = db.Column(db.Integer)
    education = db.Column(db.String(50))
    employed = db.Column(db.String(3))
    loan_amount = db.Column(db.Float)
    loan_amount_term = db.Column(db.Integer)
    residential_assets_value = db.Column(db.Float)
    commercial_assets_value = db.Column(db.Float)
    luxury_assets_value = db.Column(db.Float)
    bank_asset_value = db.Column(db.Float)


class LoanCalculation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    loan_amount = db.Column(db.Float)
    interest_rate = db.Column(db.Float)
    loan_tenure = db.Column(db.Integer)
    emi = db.Column(db.Float)

@app.route('/')
def index():
    return render_template('edu.html')

@app.route('/submit_registration', methods=['POST'])
def submit_registration():
    name = request.form['Name']
    phone = request.form['phone']
    pan = request.form['pan']
    email = request.form['login-email']
    otp = request.form['otp']
    new_user = User(name=name, phone=phone, pan=pan, email=email, otp=otp)
    db.session.add(new_user)
    db.session.commit()
    return 'Registration submitted successfully.'

@app.route('/submit_recommendation', methods=['POST'])
def submit_recommendation():
    return 'Loan recommendation submitted successfully.'

if __name__ == '__main__':
    app.run(debug=True)
