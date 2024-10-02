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

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    vision_prompt_msg = (
        f'You are the vision analysis AI that provides semantic meaning from images to provide context to send to another AI that will create a response to the user. '
        f'Do not respond as the AI assistant to the user. Instead take user prompt input and try to extract all meaning from the photo relevant to user prompt. '
        f'Then generate as much objective data about the image for the AI assistant who will respond to the user. You have spotipy library installed in you, so you can control songs too now. \nUSERPROMPT: {prompt}'
    )
    response = model.generate_content([vision_prompt_msg, img])
    return response.text

def run_assistant():
    command = take_command()

    if command:
        print(f"Command received: {command}")

        # Handle Spotify commands through Groq
        response = groq_prompt(prompt=command, img_context=None)

        # Interpret response for Spotify commands
        if 'play' in response:
            track = response.replace('play', '').strip()
            if track:
                results = sp.search(q=track, type='track')
                if results['tracks']['items']:
                    try:
                        track_uri = results['tracks']['items'][0]['uri']
                        sp.start_playback(uris=[track_uri])  # Changed to use uris
                        talk(f'Playing {results["tracks"]["items"][0]["name"]} by {results["tracks"]["items"][0]["artists"][0]["name"]}.')
                    except spotipy.exceptions.SpotifyException as e:
                        talk(f'Error occurred: {e}')
                        print(f'Error: {e}')
                else:
                    talk('Track not found.')
            else:
                talk('Please specify a track to play.')

        elif 'pause' in response:
            sp.pause_playback()
            talk('Playback paused.')

        elif 'next' in response:
            sp.next_track()
            talk('Playing next track.')

        elif 'previous' in response:
            sp.previous_track()
            talk('Playing previous track.')

        # Handle other function calls
        call = function_call(command)
        visual_context = None  

        if 'take screenshot' in call:
            print('Taking screenshot')
            take_screenshot()
            visual_context = vision_prompt(prompt=command, photo_path='screenshot.jpg')

        elif 'capture webcam' in call:
            print('Capturing webcam')
            photo_path = web_cam_capture()
            if photo_path:
                visual_context = vision_prompt(prompt=command, photo_path=photo_path)

        # Generate final response using Groq
        response = groq_prompt(prompt=command, img_context=visual_context)
        
        print(response)  
        talk(response)  
    else:
        print("No command detected.")

if __name__ == "__main__":
    while True:
        run_assistant()
