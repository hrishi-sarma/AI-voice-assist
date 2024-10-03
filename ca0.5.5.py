import speech_recognition as sr
import pyttsx3
import pyperclip
from PIL import ImageGrab, Image
import google.generativeai as genai
import cv2
import requests
from groq import Groq
import subprocess
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) 

def talk(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"User said: {command}\n")
    except Exception as e:
        print("Sorry, I did not get that.")
        return None
    return command.lower()

# Spotify API Credentials
SPOTIPY_CLIENT_ID = 'your_client_id'  # Replace with your Spotify client ID
SPOTIPY_CLIENT_SECRET = 'your_client_secret'  # Replace with your Spotify client secret
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'  # Replace with your redirect URI

# Spotify OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope='user-modify-playback-state,user-read-playback-state'))

# Initialize Groq client and generative AI model
groq_client = Groq(api_key="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")
genai.configure(api_key='AIzaSyCz8-rZ2AXUfZRRkuqHJRLXyhqBsOBnsRU')
model = genai.GenerativeModel('gemini-1.5-flash-latest')

sys_msg = (
    'You are a multi-model AI voice assistant. Your user may or may not have attached a photo for context (either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed text prompt that will be attached to their transcribed voice prompt. Generate the most useful and factual response possible, carefully considering all previous generated text in your response before adding new tokens to the response. Do not expect or request images, just use context if added. Use all of the context of this conversation so your response is relevant to the conversation. Make your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 512
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
]

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n IMAGE CONTEXT: {img_context}'
    
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will respond with only one selection from that list: ["extract clipboard", "take screenshot", "capture webcam", "None"]. Do not respond with anything but the most logical selection from that list with no explanations.'
    )
    function_convo = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

def web_cam_capture():
    web_cam = cv2.VideoCapture(0)
    if not web_cam.isOpened():
        print('Error: cam did not open')
        return None
    ret, frame = web_cam.read()
    if ret:
        path = 'webcam.jpg'
        cv2.imwrite(path, frame)
        return path
    return None

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No text')
        return None
    
def get_spotify_credentials():
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    redirect_uri = 'http://localhost:8888/callback'
    return client_id, client_secret, redirect_uri

# Function to play a song
def play_song(track_name):
    try:
        # Simplified the search query to only the track name
        results = sp.search(q=track_name, type='track')
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            sp.start_playback(uris=[track['uri']])
            talk(f"Now playing {track['name']} by {track['artists'][0]['name']}.")
        else:
            talk("Sorry, I couldn't find that track.")
    except Exception as e:
        print(f"An error occurred: {e}")
        talk("Sorry, there was an error while trying to play the song.")

# Main function to run the assistant
def run_assistant():
    while True:
        command = take_command()
        if command is None:
            continue
        if 'next song' in command:
            # Call the function to play the next song or suggest a similar song
            play_song("Electric by Alina Baraz feat. Khalid")  # You can modify this part as needed
