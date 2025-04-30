from flask import Blueprint, request, jsonify
from app.db.mongo import db
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    # Fetch training data
    training_data = list(db.chat.find({}, {"_id": 0, "input": 1, "response": 1}))
    inputs = [entry["input"] for entry in training_data]
    responses = [entry["response"] for entry in training_data]

    if not inputs:
        return jsonify({'reply': "I don't know how to respond yet. Please train me!"})

    # Vectorize all inputs + user message
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(inputs + [user_message])

    # Calculate similarity (last vector is the user input)
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_index = similarity.argmax()
    best_score = similarity[0, best_match_index]

    if best_score > 0.6:
        reply = responses[best_match_index]
    else:
        reply = "I'm still learning. Can you help me by training more?"

    return jsonify({'reply': reply})



@chat_bp.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')

    db.training.insert_one({
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })

    return jsonify({"message": "Training data saved."})


@chat_bp.route('/memory', methods=['GET'])
def memory():
    messages = list(db.memory.find({}, {'_id': 0}))
    return jsonify({"memory": messages})
