import os, re, sys
from flask import Flask, redirect, url_for, request, render_template, flash, jsonify
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
backend_dir = os.path.join(parent_dir, "backend")
sys.path.append(backend_dir)
from machineLearning import predictionModel

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "super secret key"
FILES_PATH = '../backend/database/individualFiles'

# Home Page
@app.route('/')
def index():
    return render_template('index.html', filenames=get_filenames())

# Return File History
@app.get("/")
def get_filenames():
    files = os.listdir(FILES_PATH)
    return files

# Upload Files
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
    return redirect('/'+uploaded_file.filename+'/load')

#Loading Simulation
@app.get('/<file_name>/load')
def loadMortgage(file_name):
    return render_template('loader.html', file_name=file_name)

#Generate Predicted Value
@app.get('/<file_name>/predict')
def predict(file_name):
    file_path = "../backend/database/individualFiles/" + file_name
    df = pd.read_csv(file_path)
    df['PPP'] = predictionModel.testFromUpload(file_name)
    df.to_csv(file_path)
    return jsonify("Prediction Complete!")

#View Specified Mortgage
#TODO change frontend to match test file
@app.get('/<file_name>')
def mortgageDetails(file_name):
    file_path = "../backend/database/individualFiles/" + file_name
    df = pd.read_csv(file_path)
    data = df.to_dict('records')[0]
    print(data)
    for item in ['Price', 'UPBatAcquisition', 'PPP', 'OriginationValue', 'PropertyValue', 'CurrentPropertyValue']:
        if not isinstance(data[item], str):
            data[item] = "${:,.2f}".format(data[item])
    for item in ['InterestRate', 'LTVRatio']:
        if not isinstance(data[item], str):
            data[item] = "{:.2%}".format(data[item]/100)
    ppp = "Recommended"
    return render_template('mortgageDetails.html', data=data, recommendation=ppp, file_name=file_name)

# Delete Specified File
@app.route("/delete/<file_name>")
def delete_file(file_name):
    os.remove(FILES_PATH + "/" + file_name)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
