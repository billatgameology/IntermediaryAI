import os
import google.generativeai as genai
from flask import Flask, request, jsonify

app = Flask(__name__)

genai.configure(api_key=os.environ.get("Gemini_API_Key"))

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
])

@app.route('/')
def hello_world():
    convo.send_message("Repeat my word, then add a word that starts with the last letter of my word.")
    return convo.last.text

@app.route('/chat', methods=['POST'])
def chat():
    # Get the incoming message from the POST request
    incoming_message = request.json.get('message')

    if not incoming_message:
        return jsonify({"error": "No message provided"}), 400

    # Log the incoming message using Flask's built-in logger
    app.logger.info(f'Received message: {incoming_message}')
    
    # Send the incoming message to the conversation
    response = convo.send_message(incoming_message)
    
    # Assuming convo.last.text provides the response message, or use response directly if available
    last_message = response.text

    app.logger.info(f'AI response: {last_message}')
    
    # Return the response message as JSON
    return jsonify({"response": last_message})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default port or one provided by Cloud Run environment
    app.run(debug=True, host='0.0.0.0', port=port)
