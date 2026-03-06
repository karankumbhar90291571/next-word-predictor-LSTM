import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max length
with open("maxlen.pkl", "rb") as f:
    max_len = pickle.load(f)

# Load model
model = load_model("lstm_model.h5")

# Title
st.title("Next Word Prediction using LSTM")

st.write("Type a sentence and the model will predict the next word.")

# User input
input_text = st.text_input("Enter your sentence")

def predict_next_word(text):

    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break

    return output_word


if st.button("Predict Next Word"):
    if input_text == "":
        st.warning("Please enter a sentence")
    else:
        next_word = predict_next_word(input_text)
        st.success(f"Predicted Next Word: {next_word}")