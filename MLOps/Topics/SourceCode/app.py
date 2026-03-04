import numpy as np
import flask
import pickle
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# Load Model & Kamus saat Startup
model = pickle.load(open("model.pkl", "rb"))
mapping = pickle.load(open("mapping.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        raw_data = request.form.to_dict()
        to_predict_list = []
        
        # Urutan kolom sesuaikan dengan dataset
        col_order = ['age', 'workclass', 'education', 'marital-status', 'occupation', 
                     'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 
                     'hours-per-week', 'native-country']

        for col in col_order:
            val = raw_data[col]
            if col in mapping: 
                reverse_mapping = {v: k for k, v in mapping[col].items()}
                to_predict_list.append(reverse_mapping.get(val, 0))
            else:
                to_predict_list.append(int(val))

        final_features = np.array(to_predict_list).reshape(1, 12)
        prediction = model.predict(final_features)

        res_text = 'Income >50K' if prediction[0] == 1 else 'Income <=50K'
        return render_template("result.html", prediction=res_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)