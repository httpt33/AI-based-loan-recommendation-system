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
        # ADD ALL THE INFO
    }
 
# main driver function
if __name__ == '__main__':
    app.run()