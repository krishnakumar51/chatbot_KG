from flask import Flask, request, jsonify, session
from flask_cors import CORS  # Importing flask-cors for CORS support
from qa.graph import get_qa_response  # Import the refactored function from qa.py
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set secret key for session management

# Enable CORS for all routes
CORS(app)

# Logging configuration
logging.basicConfig(level=logging.INFO)

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat messages."""
    data = request.json
    prompt = data.get('prompt')

    # Initialize conversation history if it doesn't exist
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Append user message to conversation history
    session['conversation_history'].append({"role": "human", "content": prompt})

    try:
        # Generate a response using the get_qa_response() function
        response = get_qa_response(prompt)

        # Format response to ensure Markdown link formatting
        markdown_response = f"**Response:**\n\n{response}"

        # Append assistant's response to conversation history
        session['conversation_history'].append({"role": "assistant", "content": markdown_response})

        # Return the response in JSON format, preserving Markdown
        return jsonify({"response": markdown_response})

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Endpoint to clear the conversation history."""
    # Check if conversation history exists in the session
    if 'conversation_history' in session:
        # Clear the conversation history from the session
        session.pop('conversation_history')
        return jsonify({"message": "Conversation history cleared successfully."}), 200
    else:
        return jsonify({"message": "No conversation history to clear."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Run the Flask app on port 5000
