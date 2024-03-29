import json
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Download nltk resources
nltk.download('punkt')
nltk.download('wordnet')

# Load intents from JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize training data
training_sentences = []
training_labels = []
labels = []

# Preprocess intents data
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        words = nltk.word_tokenize(pattern)
        training_sentences.append(words)
        training_labels.append(intent['tag'])
        
        # Add tag to labels list
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# Lemmatize words and convert them to lowercase
lemmatized_words = []
for word in training_sentences:
    lemmatized = [lemmatizer.lemmatize(w.lower()) for w in word]
    lemmatized_words.append(lemmatized)

# Flatten the list of lists into a single list
words_flattened = [word for sublist in lemmatized_words for word in sublist]

# Remove duplicates and sort the words
words = sorted(set(words_flattened))

# Create training data
X_train = []
y_train = []

# Create bag of words
for idx, sentence in enumerate(lemmatized_words):
    bag = []
    for word in words:
        bag.append(sentence.count(word))
    X_train.append(bag)
    
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(training_labels)

X_train = np.array(X_train)
y_train = np.array(y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def preprocess(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

def bag_of_words(sentence, words):
    sentence_words = preprocess(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    return model.predict_proba(np.array([bow]))[0]

def get_response(intent_tag):
    for intent in data['intents']:
        if intent['tag'] == intent_tag:
            return np.random.choice(intent['responses'])

app = Flask(__name__)

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/messages/", methods=["POST"])   
def get_bot_response():
    userText = request.json["message"]
    predicted_intent_prob = predict_class(userText)
    predicted_intent_idx = np.argmax(predicted_intent_prob)
    response = get_response(label_encoder.classes_[predicted_intent_idx])
    return jsonify({"message": response})

if __name__ == '__main__':
    app.run(debug=True)
