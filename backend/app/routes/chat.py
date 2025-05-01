from flask import Blueprint, request, jsonify
from app.db.mongo import db
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)

# Persist vectorizer (optional, initialize during training)
vectorizer = TfidfVectorizer()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        # Validate request data
        data = request.get_json()
        if not data or 'message' not in data or not isinstance(data['message'], str):
            return jsonify({'error': 'Invalid or missing message in request'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Fetch training data
        training_data = list(db.training.find({}, {"_id": 0, "question": 1, "answer": 1}))
        if not training_data:
            return jsonify({'reply': "I'm still learning. Please train me first."}), 200

        # Prepare inputs and responses
        inputs = [entry["question"] for entry in training_data if entry.get("question")]
        responses = [entry["answer"] for entry in training_data if entry.get("answer")]
        
        if not inputs or not responses:
            return jsonify({'reply': "No valid training data available."}), 200

        # Vectorize and compute similarity
        vectors = vectorizer.fit_transform(inputs + [user_message])
        similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])[0]  # Get 1D array

        best_match_index = similarity_scores.argmax()
        best_score = similarity_scores[best_match_index]

        logger.info(f"User message: {user_message}, Best score: {best_score}")

        # Return response based on similarity threshold
        if best_score > 0.6:
            reply = responses[best_match_index]
        else:
            reply = "I didn't quite get that. Try teaching me more!"

        return jsonify({'reply': reply}), 200

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@chat_bp.route('/train', methods=['POST'])
def train():
    try:
        # Validate request data
        data = request.get_json()
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({'error': 'Missing question or answer in request'}), 400

        question = data['question'].strip()
        answer = data['answer'].strip()

        if not question or not answer:
            return jsonify({'error': 'Question and answer cannot be empty'}), 400

        # Save to database
        db.training.insert_one({
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow()
        })

        return jsonify({"message": "Training data saved."}), 200

    except Exception as e:
        logger.error(f"Error in /train endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@chat_bp.route('/memory', methods=['GET'])
def memory():
    try:
        messages = list(db.memory.find({}, {'_id': 0}))
        return jsonify({"memory": messages}), 200
    except Exception as e:
        logger.error(f"Error in /memory endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500