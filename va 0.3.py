import pyttsx3
import re
from faster_whisper import WhisperModel
import pyperclip
from PIL import ImageGrab, Image
import cv2
import sounddevice as sd
import numpy as np
import io
import os

# Set the environment variable to bypass the OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize TTS engine
tts_engine = pyttsx3.init()

# Initialize STT model
model = WhisperModel("base")  # Replace "base" with the appropriate model you want to use

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen():
    print("Listening...")
    # Record audio
    fs = 16000  # Sample rate
    duration = 5  # Duration in seconds
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished

    # Convert audio to byte stream
    audio_stream = io.BytesIO(audio_data)
    
    # Transcribe audio
    result = model.transcribe(audio_stream)
    return result['text']

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"]. Do not respond with anything but the most logical selection from that list with no explanations.'
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
    path = 'webcam.jpg'
    if not web_cam.isOpened():
        print('Error: cam did not open')
        exit()
    ret, frame = web_cam.read()
    if ret:
        cv2.imwrite(path, frame)
    else:
        print('Error: failed to capture image')

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
        f'Then generate as much objective data about the image for the AI assistant who will respond to the user. \nUSERPROMPT: {prompt}'
    )
    response = model.generate_content([vision_prompt_msg, img])
    return response.text

web_cam = cv2.VideoCapture(0)

while True:
    prompt = listen()
    call = function_call(prompt)

    visual_context = None  # Initialize as None by default

    if 'take screenshot' in call:
        print('Taking screenshot')
        take_screenshot()
        visual_context = vision_prompt(prompt=prompt, photo_path='screenshot.jpg')

    elif 'capture webcam' in call:
        print('Capturing webcam')
        web_cam_capture()
        visual_context = vision_prompt(prompt=prompt, photo_path='webcam.jpg')

    elif 'extract clipboard' in call:
        print('Copying clipboard text')
        clipboard_content = get_clipboard_text()
        prompt = f'{prompt}\n\n CLIPBOARD CONTENT: {clipboard_content}'

    response = groq_prompt(prompt=prompt, img_context=visual_context)
    print(response)
    speak(response)
