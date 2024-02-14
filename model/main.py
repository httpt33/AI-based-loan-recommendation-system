import predict
from flask import Flask, request
from flask_cors import CORS, cross_origin

personal_model= predict.load_personal_model()
home_model= predict.load_home_model()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
 
@app.route('/')
def index():
    return 'Server is up'

@app.route('/loan/personal', methods=['POST', 'GET'])
@cross_origin()
def handle_info():
    info={
        #'Education': request.form[Education], 
        # ADD ALL THE INFO
    }
    return 'hello'
 
# main driver function
if __name__ == '__main__':
    app.run()