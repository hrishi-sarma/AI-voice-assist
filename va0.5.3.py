import speech_recognition as sr
import pyttsx3
import pyperclip
from PIL import ImageGrab, Image
import google.generativeai as genai
import cv2
import requests
from groq import Groq
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
SPOTIPY_CLIENT_ID = 'a8927d32113449d9a7f0331e442e5165'  # Replace with your Spotify client ID
SPOTIPY_CLIENT_SECRET = '550a6cafec024ea5b3f6c04caf154c5e'  # Replace with your Spotify client secret
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

def play_song(track_name):
    try:
        results = sp.search(q=track_name, type='track')
        if results['tracks']['items']:
            track_uri = results['tracks']['items'][0]['uri']
            sp.start_playback(uris=[track_uri])  # Start playback of the track URI
            track_title = results['tracks']['items'][0]['name']
            track_artist = results['tracks']['items'][0]['artists'][0]['name']
            print(f"Now playing: {track_title} by {track_artist}")
            talk(f"Now playing {track_title} by {track_artist}.")
        else:
            talk("Sorry, I couldn't find that song.")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify error occurred: {e}")
        talk("I'm sorry, I couldn't access Spotify at the moment.")

def groq_prompt(prompt):
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def get_playlist_tracks(playlist_id):
    try:
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']
        track_list = []

        for item in tracks:
            track = item['track']
            track_list.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'uri': track['uri']
            })

        return track_list
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error fetching playlist: {e}")
        return []

def play_playlist(playlist_id):
    tracks = get_playlist_tracks(playlist_id)
    if tracks:
        # Play the first track in the playlist
        sp.start_playback(uris=[tracks[0]['uri']])
        print(f"Now playing: {tracks[0]['name']} by {tracks[0]['artist']}")
        talk(f"Now playing {tracks[0]['name']} by {tracks[0]['artist']}.")
    else:
        talk("Sorry, I couldn't find any songs in that playlist.")

def run_assistant():
    command = take_command()

    if command:
        print(f"Command received: {command}")

        # Process Spotify commands through Groq
        response = groq_prompt(command)
        print("Response from Groq:", response)

        # Check if response contains Spotify control commands
        if 'play playlist' in command:
            playlist_name = command.replace('play playlist', '').strip()
            if playlist_name:
                # Assuming you know the playlist ID or can retrieve it through another method
                playlist_id = "2bXdzn8fOGPb17u7hexCTm"  # Replace with your actual playlist ID
                play_playlist(playlist_id)
            else:
                talk('Please specify a playlist to play.')

        elif 'play' in command:
            track = command.replace('play', '').strip()
            if track:
                play_song(track)
            else:
                talk('Please specify a track to play.')

        elif 'pause' in command:
            sp.pause_playback()
            talk('Playback paused.')

        elif 'next' in command:
            playback_info = sp.current_playback()
            if playback_info and playback_info['is_playing']:
                sp.next_track()
                talk('Playing next track.')
            else:
                talk('Playback is paused. Please resume playback to skip tracks.')

        elif 'previous' in command:
            sp.previous_track()
            talk('Playing previous track.')

        else:
            talk(response)  # If not a command, just talk back the response
    else:
        print("No command detected.")

if __name__ == "__main__":
    while True:
        run_assistant()
