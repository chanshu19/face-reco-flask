from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from fr_api import *
import os
app = Flask(__name__)
CORS(app)

# Creating endpoints(routes) :
@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        try:
            recieved_image = request.files['image']
            filename = secure_filename(recieved_image.filename)
            recieved_image.save(os.path.join(os.getcwd()+"/static/img",filename))
            saved_img = os.path.join(os.getcwd()+"/static/img",filename)
            return "Image saved to "+saved_img
        except Exception as e :
            print("something went wrong"+str(e))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='localhost',debug=True)
