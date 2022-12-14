from flask import Flask, request, Response, render_template
from jinja2.utils import Markup
import os
from joblib import load
# from werkzeug import secure_filename
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
PATH = os.getcwd()

def load_Model():
  clf = load(PATH + '/pickle/model.pkl')
  return clf

model = load_Model()

@app.route('/')
def index():
  return render_template('welcome.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    
    file = request.files['file']

    data = pd.read_csv(file)

    y_pred = model.predict(data)
    data['y_pred'] = y_pred

    if request.form.get("Display"):
      return render_template('result.html', data=data.to_dict(orient="records"))
    else:
      return Response(data.to_csv(), mimetype="text/csv", headers={"Content-disposition":"attachment; filename=result.csv"})    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=1234, debug=True)