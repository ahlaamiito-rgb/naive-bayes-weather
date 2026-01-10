from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

app = Flask(__name__)

# Dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Rain', 'Sunny', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','High','Normal','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Weak','Weak','Strong','Strong'],
    'Play': ['No','No','Yes','Yes','Yes','No','No','Yes','Yes','Yes']
})

X = data[['Outlook','Temperature','Humidity','Wind']]
y = data['Play']

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = CategoricalNB()
model.fit(X_encoded, y_encoded)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = pd.DataFrame([{
        'Outlook': request.form['outlook'],
        'Temperature': request.form['temperature'],
        'Humidity': request.form['humidity'],
        'Wind': request.form['wind']
    }])

    encoded = encoder.transform(input_data)
    prediction = model.predict(encoded)
    result = label_encoder.inverse_transform(prediction)[0]

    return jsonify(result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

