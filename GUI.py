import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
import altair as alt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Load the text file
with open('redit.txt', 'r') as file:
    text = file.read()

# Define a function to perform sentiment analysis using VADER
def vader_sentiment(text):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    vader_score = sia.polarity_scores(text)
    return vader_score['compound']

# Define a function to perform sentiment analysis using TextBlob
def textblob_sentiment(text):
    textblob_score = TextBlob(text).sentiment.polarity
    return textblob_score

# Define a function to perform sentiment analysis using Flair
def flair_sentiment(text):
    classifier = TextClassifier.load('en-sentiment')
    sentence = Sentence(text)
    classifier.predict(sentence)
    flair_score = sentence.labels[0].score
    return flair_score

# Define the options to switch between the models
options = ['VADER', 'TextBlob', 'Flair']

# Display the model selection dropdown
model_name = st.sidebar.selectbox('Select a model', options)

# Perform sentiment analysis using the selected model
if model_name == 'VADER':
    score = vader_sentiment(text)
elif model_name == 'TextBlob':
    score = textblob_sentiment(text)
else:
    score = flair_sentiment(text)

# Display the score
st.write(f'{model_name} score: {score}')

# Visualize the scores using a bar chart
scores = pd.DataFrame({
    'Model': options,
    'Score': [vader_sentiment(text), textblob_sentiment(text), flair_sentiment(text)]
})
chart = alt.Chart(scores).mark_bar().encode(
    x='Model',
    y='Score'
)
st.altair_chart(chart, use_container_width=True)


# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create the Streamlit web application
st.title("Sentiment Analyser for New Dataset")
text_input = st.text_input("Enter text to analyze", value="")
if st.button("Analyze"):
    # Tokenize the text and convert it to a tensor
    inputs = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")

    # Perform the sentiment analysis
    outputs = model(**inputs)
    probas = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    sentiment_label = "positive" if probas[1] > probas[0] else "negative"

    # Display the result
    st.write(f"The sentiment of the text is {sentiment_label}.")
