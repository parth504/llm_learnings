import warnings
warnings.filterwarnings("ignore")

import sys
import requests
import json
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import torch
import time
from gtts import gTTS
from playsound import playsound

# RAG specific imports
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# -------------------------------
# Configuration
# -------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3:8b"

# JSON File Paths (Make sure these exist in the same folder)
DATA_FILES = ["North_Karnataka.json", "que&ans.json"]

# ChromaDB Configuration
CHROMA_PATH = "./kannada_agri_db"
COLLECTION_NAME = "kannada_agri_knowledge"

# -------------------------------
# RAG Pipeline Class
# -------------------------------
class KannadaRAG:
    def __init__(self):
        print("📥 Loading Embedding Model (Multilingual)...")
        # Use a multilingual model that supports Kannada
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize ChromaDB Client
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Define embedding function for Chroma
        # We wrap the sentence-transformer to work with Chroma's interface
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # Create or Get Collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_func
        )
        
        # Load data if collection is empty
        if self.collection.count() == 0:
            self.load_data_to_vector_db()
        else:
            print(f"✅ Vector DB already contains {self.collection.count()} documents.")

    def load_data_to_vector_db(self):
        print("📂 Reading JSON files and indexing data...")
        documents = []
        metadatas = []
        ids = []
        id_counter = 1

        for file_path in DATA_FILES:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for entry in data:
                    # Create a rich context chunk: "Question: ... Answer: ..."
                    # This helps the model match queries to both questions and answer content
                    question = entry.get('question', '')
                    answer = entry.get('answer', '')
                    
                    if question and answer:
                        text_chunk = f"ಪ್ರಶ್ನೆ: {question}\nಉತ್ತರ: {answer}"
                        documents.append(text_chunk)
                        metadatas.append({"source": file_path, "original_q": question})
                        ids.append(f"id_{id_counter}")
                        id_counter += 1
            except FileNotFoundError:
                print(f"⚠️ Warning: File {file_path} not found. Skipping.")

        if documents:
            # Add to ChromaDB (It handles tokenization and embedding internally via the func)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"🎉 Successfully indexed {len(documents)} Q&A pairs into Vector DB.")
        else:
            print("❌ No data found to index.")

    def retrieve_context(self, query_text, n_results=3):
        """Retrieves top N relevant documents for the query."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Extract documents from results
        context_list = results['documents'][0]
        return "\n\n".join(context_list)

# -------------------------------
# Chatbot Setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading Whisper model on {device}...")
whisper_model = whisper.load_model("medium", device=device)

# Initialize RAG System
rag_system = KannadaRAG()

conversation_history = []
MAX_TURNS = 10

# Updated System Prompt to enforce RAG usage
SYSTEM_PROMPT = """
You are a helpful agricultural assistant for farmers in Karnataka.
You have access to a specific knowledge base about crops, soil, and weather in North Karnataka.

CRITICAL INSTRUCTIONS:
1. USE the provided 'Context' to answer the user's question.
2. IF the answer is found in the Context, repeat the specific details (numbers, crop names, seasons) accurately in Kannada.
3. You MUST ALWAYS reply in the KANNADA language (ಕನ್ನಡ).
4. Never reply in English.
5. Keep answers concise and helpful.

Context:
{context}
"""

def speak_kn(text):
    """Converts Kannada text to speech."""
    try:
        if not text: return
        # Clean up asterisks or special chars that might affect TTS
        clean_text = text.replace('*', '').replace('#', '')
        
        tts = gTTS(text=clean_text, lang='kn', slow=False)
        filename = f"resp_{int(time.time())}.mp3"
        tts.save(filename)
        playsound(filename)
        time.sleep(0.5)
        try:
            os.remove(filename)
        except PermissionError:
            pass
    except Exception as e:
        print(f"TTS Error: {e}")

def ask_ollama(user_query_kn: str) -> str:
    global conversation_history
    
    # 1. RETRIEVE Context from Vector DB
    print("🔍 Searching Knowledge Base...")
    context_data = rag_system.retrieve_context(user_query_kn)
    print(f"📄 Retrieved Context:\n{context_data[:200]}...") # Print first 200 chars for debug

    # 2. Inject Context into System Prompt
    current_system_prompt = SYSTEM_PROMPT.format(context=context_data)

    try:
        messages = [{"role": "system", "content": current_system_prompt}]

        # Add history
        for turn in conversation_history[-MAX_TURNS:]:
            messages.append(turn)

        messages.append({"role": "user", "content": user_query_kn})

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1} # Very low temp to force strict adherence to context
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        response_json = response.json()
        response_kn = response_json['message']['content']
        
        return response_kn

    except Exception as e:
        return f"ದೋಷ (Error): {e}"

def listen_from_mic_kn():
    print("\n🎤 ಮಾತನಾಡಲು ಪ್ರಾರಂಭಿಸಿ (Speak now)...")
    fs = 16000
    duration = 5 
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    wav.write(wav_path, fs, audio)

    try:
        result = whisper_model.transcribe(wav_path, language="kn")
        text = result["text"].strip()
    except Exception as e:
        text = ""
        print(f"Transcription Error: {e}")
    finally:
        try: os.remove(wav_path)
        except: pass
            
    return text

def main():
    global conversation_history
    print("\n🌾 ಕನ್ನಡ ಕೃಷಿ ಚಾಟ್‌ಬಾಟ್ (RAG Enabled) ಸಿದ್ಧವಿದೆ")
    print("------------------------------------------------")
    
    while True:
        try:
            mode = input("\nಮೋಡ್ ಆಯ್ಕೆ (text/voice/exit): ").lower().strip()
        except KeyboardInterrupt:
            break
            
        if mode == "exit": break
        
        user_query = ""
        if mode == "voice":
            user_query = listen_from_mic_kn()
            print(f"You said: {user_query}")
        else:
            user_query = input("ನೀವು: ").strip()

        if not user_query: continue

        print("🤖 ಆಲೋಚಿಸುತ್ತಿದೆ (Processing)...")
        response = ask_ollama(user_query)
        
        # Clean response of thinking tags if any
        response = response.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
        
        print(f"Bot: {response}")
        
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": response})
        
        speak_kn(response)

if __name__ == "__main__":
    main()