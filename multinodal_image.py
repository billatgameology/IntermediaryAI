import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def generate(base64_string):
    vertexai.init(project="project-ada-418422", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")

    # Decode the base64 string into bytes
    image_bytes = base64.b64decode(base64_string)

    # Create a Part object from the image bytes
    image_part = Part.from_data(
        mime_type="image/png",
        data=image_bytes
    )

    responses = model.generate_content(
        [image_part, """transcribe the hand written text in this photo, use symbols for arrows"""],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    transcribed_text = ""
    for response in responses:
        transcribed_text += response.text

    return transcribed_text

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}