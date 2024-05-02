import os
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

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
        return ChatGoogleGenerativeAI(model="gemini-pro")
    elif llm_model == 'Google_Gemini_1_5':
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    elif llm_model == 'Groq_Llama3_8B':
        return ChatGroq(model_name="llama3-8b-8192")
    elif llm_model == 'Groq_Llama3_70B':
        return ChatGroq(model_name="llama3-70b-8192")
    elif llm_model == 'Groq_Mistral_8x7B':
        return ChatGroq(model_name="mixtral-8x7b-32768")
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")

@tool
def OpenNewDocument(instruction: str) -> None:
    "Creates and opens a new word document"
    pass

@tool
def SummonUncleTim(greeting: str) -> None:
    "When user wants to speak with Tim, use this function"
    pass

@tool
def ChangeModel(model: str) -> str:
    "Select a LLM model, the options are Azure_GPT4, Azure_GPT3_5, Google_Gemini_1, Groq_Llama3_8B" 
    return model     

@tool
def StoryKeyOutput(who: str, what: str, when: str) -> str:
    "Extract the key output from the story, who, what, when" 
    return who + " " + what + " " + when  

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

    tools = [SummonUncleTim, OpenNewDocument]
    llm_with_tools = llm.bind_tools(tools)

    if llm is None:
        return jsonify({"error": f"LLM model not implemented: {llm_model}"}), 400

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
    chain_response = llm_with_tools.invoke(langchain_messages)

    # Check for tool calls in the response and parse accordingly
    if chain_response.tool_calls:
        tool_responses = []
        for tool_call in chain_response.tool_calls:
            tool_info = {
                'tool_name': tool_call['name'],
                'arguments': tool_call['args']
            }
            if 'output' in tool_call:
                tool_info['output'] = tool_call['output']
            tool_responses.append(tool_info)
        output = {'tool_calls': tool_responses}
    else:
        output = {'content': chain_response.content}

    # Return the response message as string
    return jsonify(output)

@app.route('/Special', methods=['POST'])
def specialist():
    # Get the incoming message from the POST request
    incoming_data = request.json

    if not incoming_data:
        return jsonify({"error": "No data provided"}), 400

    # Log the incoming data using Flask's built-in logger
    app.logger.info(f'Received data: {incoming_data}')

    # Create the LLM based on the selected model
    llm = VertexAI(model="gemini-1.5-pro-preview-0409")

    # Extract the messages from the incoming data
    messages = incoming_data.get('Messages', [])

    # Create a list to store the Langchain messages
    langchain_messages = []

    with open("_combined.md", "r") as file:
        knowledge = file.read()
    
    prompt = """
    You are representing Tim Cain, a game developer with 40 years of experience in the industry. 
    You are here to help users with their game design and development questions based on the knowledge and experience you have gathered from your video transcripts. 
    You should provide insightful and helpful answers based on the information available to you. 
    You are helping user by giving feedback or your thoughts on game design or any other aspects of game development.
    You call yourself Uncle Tim Bot. Below are all of Tim's video transcript. Your responses should be based on these transcripts.
    Do not extend principles or ideas beyond what Tim Cain has said in his videos.
    Do not answer questions that are not related to information in the transcripts.
    Do not break character or provide information that Tim Cain has not said in his videos.
    If there are information that you don't know, you can ask user to go to Tim Cain's Youtube channel and ask in the comment section.
    Provide all relavant transcript title and links used as reference in the response.
    Always provide video reference in your response.
    Your response should always be in the following format: 
    Uncle Tim Bot:
    [Response]
    List of references (at least one or more):
    [Video Title Reference] - [Youtube Link in plain text]
    """

    langchain_messages.append(SystemMessage(content=prompt))
    langchain_messages.append(HumanMessage(content=knowledge))

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
    chain_response = llm.invoke(langchain_messages)

    # Create a dictionary with the labeled response
    response_data = {'content': chain_response}

    # Return the response message as JSON
    return jsonify(response_data)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default port or one provided by Cloud Run environment
    app.run(debug=True, host='0.0.0.0', port=port)