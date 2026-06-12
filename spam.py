"""
Spam Detector Application
Flask backend with ML model and API routes
"""

from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'spam-detector-secret-key-2024'

# ==================== TRAINING DATA ====================
training_messages = [
    # Spam messages (1)
    "free offer win money now",
    "claim your free prize",
    "win cash prize click here",
    "congratulations you won",
    "urgent action required",
    "verify your account now",
    "click here to claim",
    "limited time offer",
    "earn money from home",
    "free gift card",
    "you are a winner",
    "cash bonus immediately",
    "free money instantly",
    "click this link to win",
    
    # Ham messages (0)
    "hello how are you",
    "let's meet tomorrow",
    "good morning friend",
    "what time is the meeting",
    "can you send the report",
    "thanks for your help",
    "see you at the party",
    "don't forget the deadline",
    "how was your weekend",
    "i love this weather",
    "happy birthday to you",
    "see you later",
    "looking forward to meeting",
    "thanks for your support"
]

training_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 14 spam
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 14 ham

# ==================== TRAIN MODEL ====================
vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_features=1000)
X = vectorizer.fit_transform(training_messages)

model = MultinomialNB(alpha=0.5)
model.fit(X, training_labels)

logger.info(f"Model trained successfully on {len(training_messages)} messages")
logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)} words")

# ==================== ROUTES ====================
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction from web form"""
    try:
        # Get message from form
        message = request.form.get('message', '').strip()
        
        # Validate input
        if not message:
            return jsonify({'error': 'Please enter a message to analyze'}), 400
        
        # Make prediction
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]
        probabilities = model.predict_proba(message_vector)[0]
        confidence = round(max(probabilities) * 100, 2)
        
        # Prepare result
        result = {
            'is_spam': bool(prediction == 1),
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': confidence,
            'message': message
        }
        
        logger.info(f"Prediction: {result['prediction']} (confidence: {confidence}%)")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message field'}), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Make prediction
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]
        probabilities = model.predict_proba(message_vector)[0]
        confidence = round(max(probabilities) * 100, 2)
        
        return jsonify({
            'success': True,
            'message': message,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'is_spam': bool(prediction == 1),
            'confidence': confidence
        })
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'training_samples': len(training_messages),
        'vocabulary_size': len(vectorizer.vocabulary_)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SPAM DETECTOR APPLICATION")
    print("="*60)
    print("📍 Web Interface: http://localhost:5000")
    print("📍 API Endpoint: http://localhost:5000/api/predict")
    print("📍 Health Check: http://localhost:5000/health")
    print("="*60)
    print("\n💡 Tip: Press CTRL+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)