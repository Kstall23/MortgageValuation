import os, re, sys, uuid
from flask import Flask, redirect, url_for, request, render_template, flash, jsonify
import pandas as pd

backend_dir = os.path.join(os.getcwd(), "backend/machineLearning")
sys.path.append(backend_dir)
import predictionModel
import trainingModel

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "super secret key"
FILES_PATH = 'backend/database/individualFiles'


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
        loanID = str(uuid.uuid4()).split('-')[0].upper()
        file_path = os.path.join(FILES_PATH, loanID+'.csv')
        uploaded_file.save(file_path)
    else:
        flash("Please select a valid csv file before submitting.")
        return redirect('/')
    return redirect('/' + loanID + '/load')


# Train Model
@app.get('/train')
def train():
    trainingModel.trainingClustersDriver()
    return redirect('/')


# Loading Simulation for Training
@app.get('/load')
def loadTraining():
    return render_template('trainloader.html')


@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html')


# Loading Simulation for Prediction
@app.get('/<loanID>/load')
def loadMortgage(loanID):
    return render_template('loader.html', file_name=loanID+'.csv')


# Generate Predicted Value
@app.get('/<loanID>/predict')
def predict(loanID):
    file_name = loanID + '.csv'
    file_path = os.path.join(FILES_PATH, file_name)
    df = pd.read_csv(file_path)
    df['PPP'], df['delinq'], df['appr'], df['depr'] = predictionModel.testFromUpload(file_name)
    df['loanID'] = loanID
    df.to_csv(file_path)
    return jsonify("Prediction Complete!")


# View Specified Mortgage
@app.get('/<loanID>')
def mortgageDetails(loanID):
    file_name = loanID + '.csv'
    file_path = os.path.join(FILES_PATH, file_name)
    df = pd.read_csv(file_path)
    data = df.to_dict('records')[0]
    for item in ['Price', 'UPBatAcquisition', 'PPP', 'OriginationValue', 'PropertyValue', 'CurrentPropertyValue']:
        if 'PPP' not in data.keys():
            return redirect('/' + file_name + '/load')
        if not isinstance(data[item], str):
            data[item] = "${:,.2f}".format(data[item])
    for item in ['InterestRate', 'LTVRatio']:
        if not isinstance(data[item], str):
            data[item] = "{:.2%}".format(data[item] / 100)
    return render_template('mortgageDetails.html', data=data, file_name=file_name)


# Delete Specified File
@app.route("/delete/<loanID>")
def delete_file(loanID):
    file_name = loanID + '.csv'
    os.remove(os.path.join(FILES_PATH, file_name))
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
