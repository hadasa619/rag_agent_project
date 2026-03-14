import os
import ssl
import urllib3
import gradio as gr
from dotenv import load_dotenv

# מעקף SSL לנטפרי
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

# הגדרת המודלים ב-Settings
Settings.embed_model = CohereEmbedding(
    model_name="embed-multilingual-v3.0",
    api_key=api_key
)

# שימוש במודל command-r-08-2024 (או נסי "command" אם זה נכשל)
# שימוש במודל החי והמעודכן ביותר לפי הטבלה שלך
Settings.llm = Cohere(
    model="command-a-03-2025", 
    api_key=api_key
)

# טעינת האינדקס
if not os.path.exists("./storage"):
    print("❌ שגיאה: תיקיית storage לא נמצאה!")
else:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()

def ask_bot(question, history):
    try:
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        return f"שגיאה: {e}"

demo = gr.ChatInterface(fn=ask_bot, title="סוכן ה-RAG שלי 🤖")

if __name__ == "__main__":
    demo.launch()