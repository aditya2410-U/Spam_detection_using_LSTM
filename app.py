from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the pre-trained model
model = load_model(r'C:\Users\TOSHIBA\Desktop\RNN\LSTM\Spammer\spam_classifier_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    preprocessed_text = preprocess_text(data)
    prediction = model.predict(preprocessed_text)
    predicted_class = "spam" if prediction > 0.5 else "ham"
    return jsonify({"class": predicted_class, "probability": float(prediction)})

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=150)
    return padded_sequence

if __name__ == '__main__':
    app.run(debug=True, port=8080)

