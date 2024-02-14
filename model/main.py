
import predict
from flask import Flask, request
from flask_cors import CORS, cross_origin

personal_model= predict.load_personal_model()
educational_model= predict.load_home_model()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
 
@app.route('/')
def index():
    return 'Server is up'

@app.route('/loan/personal', methods=['POST'])
@cross_origin()
def handle_info():
    info={
        #'Education': request.form[Education], 
        # ADD ALL THE INFO
    }

    return {"prediction": predict.can_get_(personal_model, predict.preprocesspersonal(info))}

@app.route('/loan/education', methods=['POST'])
@cross_origin()
def handle_info_edu():
    info={
        #'Education': request.form[Education], 
        # ADD ALL THE INFO
    }

    return {"prediction": predict.can_get_(educational_model, predict.preprocessedu(info))}
    
 
# main driver function
if __name__ == '__main__':
    app.run()