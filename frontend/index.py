import os, re
from flask import Flask, redirect, url_for, request, render_template, flash
import pandas as pd
import urllib.request, json
import certifi

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "super secret key"
TEST_API_URL = "https://api.coindesk.com/v1/bpi/currentprice.json"
FILES_PATH = '../backend/database/individualFiles'


def get_filenames():
    files = os.listdir(FILES_PATH)
    return files


@app.route('/delete/<file_name>')
def delete_file(file_name):
    os.remove(FILES_PATH + "/" + file_name)
    return redirect('/')


@app.route('/')
def index():
    return render_template('index.html', filenames=get_filenames())


@app.route("/", methods=['POST'])
def uploadFiles():
    pattern = re.compile(r'.*\.csv$')
    uploaded_file = request.files['file']
    if pattern.match(uploaded_file.filename):
        file_path = os.path.join('../backend/database/individualFiles', uploaded_file.filename)
        uploaded_file.save(file_path)
    else:
        flash("Please select a valid csv file before submitting.")
        return redirect('/')
    return redirect('/onImport/' + uploaded_file.filename)


@app.route('/onImport/<file_name>')
def mortgageDetails(file_name):
    file_path = "../backend/database/individualFiles/" + file_name
    df = pd.read_csv(file_path)
    data = df.values[0]
    price_response = urllib.request.urlopen(TEST_API_URL, cafile=certifi.where()).read()
    bitcoinValueString = json.loads(price_response)['bpi']['USD']['rate']
    bitcoinValue = int(re.sub(r'[^\w\s]', '', bitcoinValueString))
    if bitcoinValue > 23000:
        bs = True
    else:
        bs = False
    icon = "icon"
    return render_template('mortgageDetails.html', data=data, recommendation=bs, icon=icon)


if __name__ == '__main__':
    app.run(debug=True)
