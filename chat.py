import os
import ssl
import urllib3
import asyncio
import gradio as gr
from dotenv import load_dotenv

# ייבוא ה-Workflow שבנינו
from workflow import RAGAgentWorkflow

# מעקף SSL
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# יצירת מופע של ה-Workflow
agent = RAGAgentWorkflow(timeout=60)

async def ask_agent(message, history):
    """
    פונקציית העזר שתקרא ל-Workflow בצורה אסינכרונית
    """
    try:
        # הרצת ה-Workflow עם השאילתה מהממשק
        result = await agent.run(query=message)
        return str(result)
    except Exception as e:
        return f"❌ שגיאה בלתי צפויה: {str(e)}"

# יצירת ממשק Gradio מותאם ל-Async
demo = gr.ChatInterface(
    fn=ask_agent,
    title="סוכן RAG מבוסס אירועים (Event-Driven) 🤖",
    description="צפי בטרמינל כדי לראות את שלבי הוולידציה, השליפה והיצירה בזמן אמת.",
    examples=["מהו בסיס הנתונים של הפרויקט?", "מהן הנחיות ה-RTL?", "טסט"],
    cache_examples=False,
)

if __name__ == "__main__":
    print("🚀 הממשק עולה (מבוסס Workflow)...")
    demo.launch()