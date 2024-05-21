from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Loading the saved model
with open('model8.sav', 'rb') as file:
    model = pickle.load(file)

def predict_clusters(data):
    clusters = model.fit_predict(data)
    labels = ['Careless', 'Standard', 'Target', 'Careful', 'Sensible']
    return [labels[cluster] for cluster in clusters]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = []
        for i in range(5):
            annual_income = float(request.form[f'sample_{i+1}_annual_income'])
            spending_score = float(request.form[f'sample_{i+1}_spending_score'])
            data.append([annual_income, spending_score])

        labels = predict_clusters(data)
        return render_template('result.html', labels=labels)

if __name__ == '__main__':
    app.run(debug=True)
