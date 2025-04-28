from flask import Flask, request, jsonify
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["ritu"] 

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # (optional) Save message to MongoDB in future
    # db.chats.insert_one({"message": message})

    return jsonify({'reply': f'You said: {message}'})

if __name__ == '__main__':
    app.run(debug=True)
