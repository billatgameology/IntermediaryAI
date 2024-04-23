import os
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

app = Flask(__name__)

llm = ChatOpenAI()

@app.route('/chat', methods=['POST'])
def chat():
    # Get the incoming message from the POST request
    incoming_data = request.json

    if not incoming_data:
        return jsonify({"error": "No data provided"}), 400

    # Log the incoming data using Flask's built-in logger
    app.logger.info(f'Received data: {incoming_data}')

    # Extract the messages from the incoming data
    messages = incoming_data.get('Messages', [])

    # Create a list to store the Langchain messages
    langchain_messages = []

    # Convert the messages to Langchain message types
    for message in messages:
        role = message['role']
        content = message['content']

        if role == 'User':
            langchain_messages.append(HumanMessage(content=content))
        elif role == 'Assistant':
            langchain_messages.append(AIMessage(content=content))
        elif role == 'System':
            langchain_messages.append(SystemMessage(content=content))

    # Generate a response using the LLM
    chain_response = llm(langchain_messages)

    # Return the response message as string
    return chain_response.content

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default port or one provided by Cloud Run environment
    app.run(debug=True, host='0.0.0.0', port=port)