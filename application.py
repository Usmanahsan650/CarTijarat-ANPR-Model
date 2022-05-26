import os;
from flask import Flask
from flask import request
from deploy import main;
import requests
from flask_cors import CORS
application = Flask(__name__)
CORS(application)
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = {'png','jpg', 'jpeg'}
def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 
@application.route('/api/anpr', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if request.files['image']!=None:
            file = request.files['image'].read()
            filename=request.files['image'].filename
            if filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                return listToString(main(img=file)) or "Oops! cant detect the license plate";
            else:
                return "file type not supported!"
        else:
            return "Invalid request no file selected!"

application.run(host='0.0.0.0', port=5000)