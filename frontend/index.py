import os, re, sys
from flask import Flask, redirect, url_for, request, render_template, flash
import pandas as pd
sys.path.append(os.path.abspath(os.pardir)+'/backend/machineLearning')
import demoPrediction

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "super secret key"
FILES_PATH = '../backend/database/individualFiles'


def get_filenames():
    files = os.listdir(FILES_PATH)
    return files


@app.route('/')
def index():
    return render_template('index.html', filenames=get_filenames())


@app.post("/")
def uploadFiles():
    pattern = re.compile(r'.*\.csv$')
    uploaded_file = request.files['file']
    if pattern.match(uploaded_file.filename):
        file_path = os.path.join('../backend/database/individualFiles', uploaded_file.filename)
        uploaded_file.save(file_path)
    else:
        flash("Please select a valid csv file before submitting.")
        return redirect('/')
    return redirect('/mortgage/' + uploaded_file.filename)


@app.route("/delete/<file_name>")
def delete_file(file_name):
    os.remove(FILES_PATH + "/" + file_name)
    return redirect('/')

@app.route('/mortgage')
def createMortgage():
    output1 = demoPrediction.predict()
    print(output1)
    return render_template('createMortgage.html', title=output1)
@app.route('/mortgage/<file_name>')
def mortgageDetails(file_name):
    file_path = "../backend/database/individualFiles/" + file_name
    df = pd.read_csv(file_path)
    data = df.values[0]
    if True:
        ppp = "Recommended"
    else:
        ppp = "Not Recommended"
    return render_template('mortgageDetails.html', data=data, recommendation=ppp, file_name=file_name)

if __name__ == '__main__':
    app.run(debug=True)
