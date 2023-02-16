import os, re
from flask import Flask, redirect, url_for, request, render_template, flash
import pandas as pd
import urllib.request, json
import certifi

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "super secret key"
TEST_API_URL = "https://api.coindesk.com/v1/bpi/currentprice.json"


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join('../backend/database/individualFiles', uploaded_file.filename)
        uploaded_file.save(file_path)
    else:
        flash("Please select a file before submitting.")
        return redirect('/')
    return redirect('/onImport/' + uploaded_file.filename)


@app.route('/onImport/<file_name>')
def mortageDetails(file_name):
    file_path = "../backend/database/individualFiles/" + file_name
    df = pd.read_csv(file_path)
    data = df.values[0]
    price_response = urllib.request.urlopen(TEST_API_URL, cafile=certifi.where()).read()
    ppp = json.loads(price_response)['bpi']['USD']['rate']
    return render_template('mortgageDetails.html', data=data, ppp=ppp)


if __name__ == '__main__':
    app.run(debug=True)
