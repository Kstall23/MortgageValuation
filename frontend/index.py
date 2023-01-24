import os

from flask import Flask, redirect, url_for, request, render_template, flash
import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "super secret key"

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join('static/files', uploaded_file.filename)
        uploaded_file.save(file_path)
    else:
        flash("Please select a file before submitting.")
        return redirect('/')
    return redirect('/onImport/'+ uploaded_file.filename)

@app.route('/onImport/<file_name>')
def mortageDetails(file_name):
    file_path = "static/files/" + file_name
    df = pd.read_csv(file_path)
    data = df.values[0]
    return render_template('mortgageDetails.html', data=data, ppp="$200,000,000")

if __name__ == '__main__':
    app.run(debug=True)
