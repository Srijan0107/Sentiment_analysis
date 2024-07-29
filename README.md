# Movie Sentiment Analysis

This project is a sentiment analysis application for movie reviews using a deep learning model. It uses an LSTM (Long Short-Term Memory) neural network to classify movie reviews as either positive or negative.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Predictive System](#predictive-system)
- [Gradio Interface](#gradio-interface)
- [Acknowledgments](#acknowledgments)

## Overview
This project utilizes the IMDB movie reviews dataset to train a sentiment analysis model. The model is built using TensorFlow and Keras, with the LSTM layer being used to handle the sequential nature of the text data. The project also includes a web interface using Gradio for users to input movie reviews and get sentiment predictions.

## Dataset
The dataset used is the IMDB movie reviews dataset, which contains 50,000 movie reviews labeled as positive or negative.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- TensorFlow
- Keras
- joblib
- Gradio

## Installation
To install the required libraries, you can use pip:
```bash
pip install pandas numpy scikit-learn tensorflow keras joblib gradio

Project Structure

mathematica
Copy code
movie-sentiment-analysis/
│
├── IMDB Dataset.csv
├── model.h5
├── tokenizer.pkl
├── README.md
├── train_model.py
├── evaluate_model.py
├── predict.py
├── app.py
└── utils.py
Usage

Data Preprocessing: Load and preprocess the dataset, including tokenizing and padding sequences.
Model Training: Build and train the LSTM model on the training data.
Model Evaluation: Evaluate the trained model on the test data.
Predictive System: Define a function to make predictions on new movie reviews.
Gradio Interface: Create a web interface for users to input reviews and get sentiment predictions.
Model Training

The model is built using an embedding layer followed by an LSTM layer and a dense output layer. The model is compiled with the Adam optimizer and binary cross-entropy loss function, and trained for 5 epochs with a batch size of 64.

python
Copy code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)

model.save("model.h5")
Model Evaluation

Evaluate the model on the test set to determine its performance.

python
Copy code
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
Predictive System

Define a function to predict the sentiment of new reviews using the trained model.

python
Copy code
def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment
Gradio Interface

Create a web interface using Gradio to allow users to input reviews and get sentiment predictions.

python
Copy code
import gradio as gr

title = "MOVIE SENTIMENT ANALYSIS APPLICATION"
app = gr.Interface(fn=predictive_system, inputs="textbox", outputs="textbox", title=title)
app.launch(share=True)
Acknowledgments

IMDB Dataset
TensorFlow
Keras
Gradio
