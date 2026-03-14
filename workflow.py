import os
import ssl
import urllib3
import asyncio
from typing import Any, List, Optional
from dotenv import load_dotenv

# --- הגדרות אבטחה וסינון (NetFree) ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, Workflow, step, Context
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere

load_dotenv()

# --- 1. הגדרת אירועים (Events) ---
class RetrievalEvent(Event):
    """אירוע המועבר לאחר ולידציית קלט מוצלחת"""
    query: str

class GenerationEvent(Event):
    """אירוע המועבר לאחר שליפת נתונים מוצלחת מה-Storage"""
    context_str: str
    query: str

class ErrorEvent(Event):
    """אירוע לטיפול במקרי קצה ושגיאות לוגיות"""
    message: str

# --- 2. הגדרת ה-Workflow ---
class RAGAgentWorkflow(Workflow):
    
    @step
    async def validate_query(self, ev: StartEvent) -> RetrievalEvent | ErrorEvent:
        """שלב 1: בדיקת תקינות השאילתה (Guardrail)"""
        query = ev.get("query")
        print(f"🛡️  שלב 1: ולידציה - בודק שאילתה: '{query}'")
        
        if not query or len(query.strip()) < 5:
            return ErrorEvent(message="השאילתה קצרה מדי. אנא נסחי שאלה מפורטת יותר.")
        
        return RetrievalEvent(query=query)

    @step
    async def retrieve_data(self, ev: RetrievalEvent) -> GenerationEvent | ErrorEvent:
        """שלב 2: שליפת מידע רלוונטי מהאינדקס המקומי"""
        print("📂 שלב 2: שליפת מידע מה-Storage המקומי...")
        
        if not os.path.exists("./storage"):
            return ErrorEvent(message="תיקיית storage לא נמצאה. יש להריץ את main.py תחילה.")

        # טעינת האינדקס מהדיסק
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        
        # הגדרת המחלץ (Retriever)
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(ev.query)
        
        # בדיקה: האם נמצא משהו?
        if not nodes:
            return ErrorEvent(message="לא נמצא מידע תואם במסמכי הפרויקט.")
        
        # בדיקת רלוונטיות (Confidence Score)
        # ב-LlamaIndex, ה-score מייצג דמיון קוסינוס. 0.35 הוא סף סביר לסינון "רעש".
        score = nodes[0].score if nodes[0].score is not None else 1.0
        if score < 0.35:
            return ErrorEvent(message=f"רמת הרלוונטיות של המידע שנמצא נמוכה מדי ({score:.2f}).")

        print(f"✅ נמצא מידע רלוונטי (Score: {score:.2f})")
        context_text = "\n\n".join([n.get_content() for n in nodes])
        return GenerationEvent(context_str=context_text, query=ev.query)

    @step
    async def generate_response(self, ev: GenerationEvent) -> StopEvent:
        print("🧠 שלב 3: ניסוח תשובה (Chat API המעודכן)...")
        
        import cohere
        co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        
        try:
            # שימוש ישיר ב-Chat API של Cohere - זה לא יכול לתת 404 של Generate
            response = co.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "אתה עוזר טכני ממוקד אך ורק לפרויקט ניהול המשימות. "
                        "תפקידך לענות על שאלות בהסתמך על המידע המסופק בלבד. "
                        "אם השאלה אינה קשורה לפרויקט, לתיעוד שלו או להחלטות הארכיטקטוניות שלו, "
                        "ענה בנימוס שאינך מוסמך לענות על נושאים מחוץ לפרויקט וסיים את התשובה."
                        "לדוגמא, אם נשאלת שאלה בנושא כללי כגון: איזו תרופה מתאימה לכאבי ראש? תגיב: 'אני מתמחה רק בנושאי הפרויקט, אנא שאל שאלה הקשורה לניהול המשימות או לתיעוד שלו.'"
                    )
                },
                {
                    "role": "user", 
                    "content": f"המידע מהתיעוד:\n{ev.context_str}\n\nהשאלה: {ev.query}"
                }
            ]
        )
            
            # שליפת הטקסט מתוך האובייקט שהחזיר ה-API
            answer = response.message.content[0].text
            return StopEvent(result=str(answer))
            
        except Exception as e:
            return StopEvent(result=f"שגיאה בקריאה ל-Cohere: {str(e)}")
    @step
    async def handle_errors(self, ev: ErrorEvent) -> StopEvent:
        """שלב חלופי: טיפול בשגיאות ודיווח למשתמש"""
        print(f"⚠️  עצירה בגלל שגיאה: {ev.message}")
        return StopEvent(result=f"מצטער, חלה שגיאה בתהליך: {ev.message}")

# --- 3. הגדרות מודלים (Global Settings) ---
def setup_settings():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("אנא הגדירי COHERE_API_KEY בקובץ .env")

    # מודל ה-Embedding
    Settings.embed_model = CohereEmbedding(
        model_name="embed-multilingual-v3.0", 
        api_key=api_key
    )

    # מודל השפה (חייב להיות מהדור החדש שתומך ב-Chat API)
    Settings.llm = Cohere(
        model="command-a-03-2025", 
        api_key=api_key
    )

setup_settings()

# --- 4. פונקציית הרצה לבדיקה עצמאית ---
if __name__ == "__main__":
    async def test_workflow():
        # יצירת מופע של ה-Agent
        agent = RAGAgentWorkflow(timeout=60)
        
        print("--- בדיקה 1: שאלה תקינה ---")
        res = await agent.run(query="איזה בסיס נתונים נבחר?")
        print(f"תשובה: {res}\n")
        
        print("--- בדיקה 2: שאילתה קצרה מדי ---")
        res_err = await agent.run(query="מה?")
        print(f"תשובה: {res_err}\n")

    asyncio.run(test_workflow())