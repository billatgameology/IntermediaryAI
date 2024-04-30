import os
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_vertexai import VertexAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

app = Flask(__name__)

def create_llm(llm_model):
    if llm_model == 'Azure_GPT4_Vision':
        return AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_GPT4_DEPLOYMENT_NAME"],
        )
        #TODO: need to be updated
    elif llm_model == 'Azure_GPT4':
        return AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_GPT4_DEPLOYMENT_NAME"],
        )
    elif llm_model == 'Azure_GPT3_5':
        return AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_GPT3_DEPLOYMENT_NAME"],
        )
    elif llm_model == 'OpenAI_GPT4':
        return ChatOpenAI(model_name='gpt-4')
    elif llm_model == 'OpenAI_GPT3_5':
        return ChatOpenAI(model_name='gpt-3.5-turbo')
    elif llm_model == 'Google_Gemini_1':
        return VertexAI(model_name="gemini-pro")
    elif llm_model == 'Google_Gemini_1_5':
        return VertexAI(model="gemini-1.5-pro-preview-0409")
    elif llm_model == 'Groq_Llama3_8B':
        return ChatGroq(model_name="llama3-8b-8192")
    elif llm_model == 'Groq_Llama3_70B':
        return ChatGroq(model_name="llama3-70b-8192")
    elif llm_model == 'Groq_Mistral_8x7B':
        return ChatGroq(model_name="mixtral-8x7b-32768")
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")
      

@app.route('/chat', methods=['POST'])
def chat():
    # Get the incoming message from the POST request
    incoming_data = request.json

    if not incoming_data:
        return jsonify({"error": "No data provided"}), 400

    # Log the incoming data using Flask's built-in logger
    app.logger.info(f'Received data: {incoming_data}')

    # Extract the LLM model from the incoming data
    llm_model = incoming_data.get('LlmModel')

    # Create the LLM based on the selected model
    llm = create_llm(llm_model)

    if llm is None:
        return jsonify({"error": f"LLM model not implemented: {llm_model}"}), 400

    # Extract the messages from the incoming data
    messages = incoming_data.get('Messages', [])

    # Create a list to store the Langchain messages
    langchain_messages = []

    # Convert the messages to Langchain message types
    # Forcing messages alternate between AI and Human for Gemini models
    previous_role = None

    for message in messages:
        role = message['role']
        content = message['content']

        if role == 'User':
            if previous_role == 'User':
                # Handle the case where a User message follows another User message
                langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
                previous_role = 'User'
        elif role == 'Assistant':
            if previous_role == 'User':
                langchain_messages.append(AIMessage(content=content))
                previous_role = 'Assistant'
            else:
                # Handle the case where an Assistant message follows a System or another Assistant message
                langchain_messages.append(HumanMessage(content=''))
                langchain_messages.append(AIMessage(content=content))
                previous_role = 'Assistant'
        elif role == 'System':
            if previous_role == 'User':
                langchain_messages.append(AIMessage(content=content))
                previous_role = 'System'
            else:
                # Handle the case where a System message follows an Assistant or another System message
                langchain_messages.append(HumanMessage(content=''))
                langchain_messages.append(AIMessage(content=content))
                previous_role = 'System'
        # Generate a response using the LLM
        chain_response = llm(langchain_messages)

        # Return the response message as string
        return chain_response.content

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default port or one provided by Cloud Run environment
    app.run(debug=True, host='0.0.0.0', port=port)