import os
import ssl
import urllib3
from dotenv import load_dotenv

# ביטול SSL בגלל החסימות
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()

def run_ingestion():
    # 1. טעינת המסמכים
    print("📂 טוען מסמכים...")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()

    # 2. הגדרת מודל ה-Embedding (זה עובד כי Cohere פתוח בנטפרי)
    embed_model = CohereEmbedding(
        model_name="embed-multilingual-v3.0",
        api_key=os.getenv("COHERE_API_KEY")
    )

    # 3. יצירת האינדקס ושמירה מקומית (עוקף את Pinecone)
    if not os.path.exists("./storage"):
        print("🧠 יוצר אינדקס חדש (מקומי)...")
        index = VectorStoreIndex.from_documents(
            documents, 
            embed_model=embed_model,
            show_progress=True
        )
        # שמירה לדיסק כדי שלא נצטרך לשלם שוב על Embeddings
        index.storage_context.persist(persist_dir="./storage")
        print("💾 האינדקס נשמר בתיקיית storage.")
    else:
        print("📚 טוען אינדקס קיים מהדיסק...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context, embed_model=embed_model)

    print("✅ המערכת מוכנה!")
    return index

if __name__ == "__main__":
    run_ingestion()