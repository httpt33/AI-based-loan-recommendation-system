from ..model import predict
from flask import Flask

personal_model= predict.load_personal_model()
home_model= predict.load_home_model()
 

app = Flask(__name__)
 
@app.route('/')
def index():
    return 'Server is up'

@app.route('/loan/personal', methods=['POST'])
def handle_info():
    info={
        'Education': request.form[Education],
        'Self_Employed': request.form[Self_Employed],
        'Loan_Amount': request.form[Loan_Amount],
        'Loan_Amount_Term': request.form[Loan_Amount_Term],
        'Cibil_Score': request.form[Cibil_Score],
        'Residential_Assets_Value': request.form[Residential_Assets_Value],
        'Commercial_Assets_Value': request.form[Commercial_Assets_Value],
        'Luxury_Assets_Value': request.form[Luxury_Assets_Value],
        'Bank_Asset_Value': request.form[Bank_Asset_Value]
        
        # ADD ALL THE INFO
    }
 
# main driver function
if __name__ == '__main__':
    app.run()