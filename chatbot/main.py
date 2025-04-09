from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from .utils import analyze_sentiment
import json
import nltk

nltk.download('vader_lexicon')
# Initialize FastAPI app
app = FastAPI()

# Initialize the chatbot model (GPT-2 or any other model)
chatbot = pipeline('text-generation', model='gpt2')

# Load predefined responses from the responses.json file
def load_responses():
    with open('data/responses.json', 'r') as file:
        return json.load(file)

responses = load_responses()

# Request model for user input
class UserInput(BaseModel):
    text: str

# Endpoint for getting a response from the chatbot
@app.post("/chat/")
async def chat(user_input: UserInput):
    user_text = user_input.text
    sentiment = analyze_sentiment(user_text)
    response = get_chatbot_response(user_text)

    # If the sentiment indicates a certain emotion, return a predefined response
    if sentiment['compound'] < -0.5:
        if "sad" in user_text.lower():
            return {"response": responses.get("feeling_down", "I'm here for you.")}
        elif "anxious" in user_text.lower():
            return {"response": responses.get("feeling_anxious", "It's okay to feel anxious. Try some deep breaths.")}

    return {"response": response}

# Function to generate a response using the chatbot model
def get_chatbot_response(text):
    response = chatbot(text, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']
