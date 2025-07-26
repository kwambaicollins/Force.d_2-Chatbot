import json
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import os
from typing import List, Dict, Tuple, Any
from functools import lru_cache

# Constants
INTENTS_FILE = 'intents.json'
MODEL_FILE = 'chatbot_model.pkl'
WORDS_FILE = 'words.pkl'
LABELS_FILE = 'labels.pkl'

# Download nltk resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class ChatbotModel:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.model = None
        self.words = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.intents = []
        self.vectorizer = None

    def load_intents(self, filepath: str) -> Dict[str, Any]:
        """Load intents from JSON file"""
        try:
            with open(filepath, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Intents file not found at {filepath}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in intents file")

    def preprocess_sentence(self, sentence: str) -> List[str]:
        """Tokenize and lemmatize a sentence"""
        words = nltk.word_tokenize(sentence)
        return [self.lemmatizer.lemmatize(word.lower()) for word in words]

    def prepare_training_data(self, intents_data: Dict[str, Any]) -> Tuple[List[List[str]], List[str]]:
        """Prepare training sentences and labels from intents"""
        training_sentences = []
        training_labels = []
        labels = []

        for intent in intents_data['intents']:
            for pattern in intent['patterns']:
                words = self.preprocess_sentence(pattern)
                training_sentences.append(words)
                training_labels.append(intent['tag'])
                
                if intent['tag'] not in labels:
                    labels.append(intent['tag'])

        return training_sentences, training_labels, labels

    def create_bag_of_words(self, sentences: List[List[str]]) -> List[List[int]]:
        """Create bag of words representation"""
        # Flatten and get unique words
        words_flattened = [word for sublist in sentences for word in sublist]
        self.words = sorted(set(words_flattened))
        
        # Create bag of words
        X_train = []
        for sentence in sentences:
            bag = [0] * len(self.words)
            for word in sentence:
                if word in self.words:
                    bag[self.words.index(word)] += 1
            X_train.append(bag)
            
        return X_train

    def train_model(self, X_train: List[List[int]], y_train: List[int]) -> None:
        """Train the classification model"""
        self.model = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.model.fit(X_train, y_train)

    def initialize(self, intents_file: str) -> None:
        """Initialize the chatbot model"""
        # Load and prepare data
        intents_data = self.load_intents(intents_file)
        self.intents = intents_data['intents']
        
        training_sentences, training_labels, self.labels = self.prepare_training_data(intents_data)
        
        # Create training data
        X_train = self.create_bag_of_words(training_sentences)
        self.label_encoder.fit(training_labels)
        y_train = self.label_encoder.transform(training_labels)
        
        # Train model
        self.train_model(X_train, y_train)

    def save_model(self, model_file: str, words_file: str, labels_file: str) -> None:
        """Save the trained model and vocabulary"""
        with open(model_file, 'wb') as f:
            pickle.dump((self.model, self.label_encoder), f)
        with open(words_file, 'wb') as f:
            pickle.dump(self.words, f)
        with open(labels_file, 'wb') as f:
            pickle.dump(self.labels, f)

    def load_saved_model(self, model_file: str, words_file: str, labels_file: str) -> bool:
        """Load saved model if available"""
        if (os.path.exists(model_file) and 
            os.path.exists(words_file) and 
            os.path.exists(labels_file)):
            try:
                with open(model_file, 'rb') as f:
                    self.model, self.label_encoder = pickle.load(f)
                with open(words_file, 'rb') as f:
                    self.words = pickle.load(f)
                with open(labels_file, 'rb') as f:
                    self.labels = pickle.load(f)
                return True
            except Exception as e:
                print(f"Error loading saved model: {e}")
                return False
        return False

    @lru_cache(maxsize=1000)
    def predict_intent(self, sentence: str) -> Tuple[str, float]:
        """Predict intent for a given sentence with caching"""
        bow = self._create_single_bow(sentence)
        if bow is None:
            return "unknown", 0.0
            
        probabilities = self.model.predict_proba(np.array([bow]))[0]
        predicted_idx = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        # Apply confidence threshold
        if confidence < 0.5:
            return "unknown", confidence
            
        return self.label_encoder.classes_[predicted_idx], confidence

    def _create_single_bow(self, sentence: str) -> List[int]:
        """Create bag of words for a single sentence"""
        sentence_words = self.preprocess_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            if s in self.words:
                bag[self.words.index(s)] = 1
        return bag

    def get_response(self, intent_tag: str) -> str:
        """Get random response for the given intent"""
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                return np.random.choice(intent['responses'])
        return "I'm not sure how to respond to that."

# Initialize Flask app and chatbot
app = Flask(__name__)
chatbot = ChatbotModel()

# Try to load saved model, otherwise train new one
if not chatbot.load_saved_model(MODEL_FILE, WORDS_FILE, LABELS_FILE):
    print("Training new model...")
    chatbot.initialize(INTENTS_FILE)
    chatbot.save_model(MODEL_FILE, WORDS_FILE, LABELS_FILE)
    print("Model trained and saved.")
else:
    print("Loaded pre-trained model.")

@app.route("/")
def home():
    """Render the chat interface"""
    return render_template("index.html")

@app.route("/messages/", methods=["POST"])   
def get_bot_response():
    """Handle chat messages and return bot response"""
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"message": "Please type something."}), 400
            
        intent, confidence = chatbot.predict_intent(user_message)
        response = chatbot.get_response(intent)
        
        # Log the interaction (in a real app, you'd want to persist this)
        app.logger.info(f"User: {user_message} | Intent: {intent} (Confidence: {confidence:.2f}) | Response: {response}")
        
        return jsonify({"message": response})
    
    except Exception as e:
        app.logger.error(f"Error processing message: {e}")
        return jsonify({"message": "Sorry, I encountered an error processing your request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
