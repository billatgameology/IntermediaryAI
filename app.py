import os
from flask import Flask

app = Flask(__name__)




import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models


def multiturn_generate_content():
  vertexai.init(project="project-ada-418422", location="us-central1")
  model = GenerativeModel(
    "gemini-1.0-pro-001",
  )
  chat = model.start_chat()
  print(chat.send_message(
      ["""hello"""],
      generation_config=generation_config,
      safety_settings=safety_settings
  ))


generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.9,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


@app.route('/')
def hello_world():
    return multiturn_generate_content()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default port or one provided by Cloud Run environment
    app.run(debug=True, host='0.0.0.0', port=port)
