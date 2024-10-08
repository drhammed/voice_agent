import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import pyaudio
import sounddevice as sd
import numpy as np
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_community.cache import RedisCache
from langchain.globals import set_llm_cache
import redis

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakOptions,
)

import wave
import json
import hashlib
from datetime import datetime


load_dotenv()


GROQ_API_KEY = os.getenv("My_Groq_API_key")
DG_API_KEY = os.getenv("DeepGram_API_key")

if not GROQ_API_KEY:
    raise EnvironmentError("Groq API key is missing, please add the API...")

if not DG_API_KEY:
    raise EnvironmentError("Deepgram API key is missing, , please add the API")



class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.02, model_name="gemma-7b-it", groq_api_key=GROQ_API_KEY)
        
        # Set up a Redis client- this was set up in Docker
        self.redis_client = redis.Redis.from_url(url="redis://default:admin@localhost:6379/0")
        # Initialize the Redis cache with the Redis client
        set_llm_cache(RedisCache(self.redis_client))
        
       # Initialize JSON cache
        self.json_cache_file = 'backup_cache.json'
        if not os.path.exists(self.json_cache_file):
            with open(self.json_cache_file, 'w') as f:
                json.dump({}, f)
               
        #Use the conversation buffer memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        system_prompt = """
        You are a helpful assistant named Ella designed to assist customers with their questions and concerns regarding their products. 
        Your role is to provide clear, concise, and accurate information based on the customer's query.  
        Use short, conversational responses as if you're having a live conversation, and your response should be under 20 words. 
        Do not respond with any code, only conversation.
        If you do not have the necessary information, politely inform the customer that the data is not available, but you can continue to engage in 
        a friendly conversation to make them feel welcomed and comfortable. Remember to keep the interactions brief and relevant 
        to the user's query.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def generate_id(self, text):
        return hashlib.sha256(text.encode()).hexdigest()
    
    def process(self, text):
        print(f"\nProcessing query: {text}")
        # Add user message to memory
        self.memory.chat_memory.add_user_message(text)
        # Get start time
        start_time = time.time()
        # Generate a unique ID for the query
        query_id = self.generate_id(text)
        
        # Check Redis cache
        cached_response = self.redis_client.get(query_id)
        if cached_response is not None:
            print("Using Redis cached response")
            return cached_response.decode('utf-8')
        
        # Check JSON cache
        with open(self.json_cache_file, 'r') as f:
            json_cache = json.load(f)
        if query_id in json_cache:
            print("Using JSON cached response")
            return json_cache[query_id]['response']
        
        # If not in either cache, use LLM
        print("No cache hit. Querying LLM...")
        response = self.conversation.invoke({"text": text})
        end_time = time.time()
        
        # Add AI response to memory
        self.memory.chat_memory.add_ai_message(response['text'])  
        
        # Cache the response in Redis
        self.redis_client.set(query_id, response['text'].encode('utf-8'))
        print("Cached response in Redis")
        
        # Cache the response in JSON
        with open(self.json_cache_file, 'r+') as f:
            json_cache = json.load(f)
            json_cache[query_id] = {
                'response': response['text'],
                'timestamp': datetime.now().isoformat()
            }
            f.seek(0)
            json.dump(json_cache, f, indent=2)
            f.truncate()
        print("Cached response in JSON")
        
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM response time: {elapsed_time}ms")
        print(f"LLM response: {response['text']}")
        return response['text']       
        

class TextToSpeech:
    def __init__(self):
        self.DG_API_KEY = DG_API_KEY
        self.MODEL_NAME = "aura-orpheus-en"  

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")
        
        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model {self.MODEL_NAME}&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        
        # To keep track- record the time before sending the request
        start_time = time.time()
        # Also, initialize a variable to store the time when the first byte is received  
        first_byte_time = None  

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    # Check if this is the first chunk received
                    if first_byte_time is None: 
                        # Record the time when the first byte is received
                        first_byte_time = time.time()  
                        # Calculate the time to first byte
                        ttfb = int((first_byte_time - start_time) * 1000)  
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    # Event to signal transcription completion
    transcription_complete = asyncio.Event()  

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(DG_API_KEY, config)

        dg_connection = deepgram.listen.asyncwebsocket.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)
                    transcript_collector.reset()
                    transcription_complete.set()

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()

        microphone.finish()
        await dg_connection.finish()
    
    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            await get_transcript(handle_full_sentence)
            
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            self.transcription_response = ""
    
if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
    