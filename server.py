from flask import Flask
from flask import request
from deploy import main;
import requests
app = Flask(__name__)
from werkzeug.utils import secure_filename

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        file.save(f"./output/car3.jpg");
        return main(img_path="./output/car3.jpg");