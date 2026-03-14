import os
import json
import ssl
import urllib3
from typing import Literal
from dotenv import load_dotenv

# --- הגדרות אבטחה (NetFree) ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, Workflow, step
)
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.base.llms.types import ChatMessage

load_dotenv()

# --- 1. הגדרת אירועים (Events) ---
class RoutingEvent(Event):
    choice: Literal["semantic", "structured"]
    query: str

class RetrievalEvent(Event):
    query: str

class StructuredRetrievalEvent(Event):
    query: str

class GenerationEvent(Event):
    context_str: str
    query: str

# --- 2. הגדרת ה-Workflow ---
class RAGAgentWorkflow(Workflow):
    
    @step
    async def router(self, ev: StartEvent) -> RoutingEvent:
        """מזהה האם השאלה היא רשימתית (JSON) או הבנתית (Semantic)"""
        query = ev.get("query")
        print(f"🚦 נתב: מנתח את השאלה: '{query}'")
        
        # פרומפט משוריין עם דוגמאות (Few-Shot)
        system_prompt = (
            "You are a routing expert for a technical RAG system.\n"
            "Categorize the query into 'structured' or 'semantic' based on these examples:\n\n"
            "Example 1: 'תן לי את כל הכללים' -> structured\n"
            "Example 2: 'אילו החלטות התקבלו?' -> structured\n"
            "Example 3: 'רשימת אזהרות' -> structured\n"
            "Example 4: 'איך עובד ה-DB?' -> semantic\n"
            "Example 5: 'למה בחרנו ב-React?' -> semantic\n\n"
            "Rules:\n"
            "- If the user asks for a LIST, ALL, SUMMARY, or EVERYTHING of a category -> structured.\n"
            "- If the user asks for EXPLANATION, WHY, or HOW -> semantic.\n"
            "Respond ONLY with 'structured' or 'semantic'."
        )
        
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=f"Query: {query}")
        ]
        
        response = await Settings.llm.achat(messages)
        choice = response.message.content.strip().lower()
        
        # ניקוי אגרסיבי של התשובה
        if "structured" in choice:
            selected = "structured"
        else:
            selected = "semantic"
            
        print(f"🛣️  נבחר מסלול: {selected}")
        return RoutingEvent(choice=selected, query=query)

    @step
    async def handle_routing(self, ev: RoutingEvent) -> RetrievalEvent | StructuredRetrievalEvent:
        if ev.choice == "structured":
            return StructuredRetrievalEvent(query=ev.query)
        return RetrievalEvent(query=ev.query)

    @step
    async def retrieve_from_json(self, ev: StructuredRetrievalEvent) -> GenerationEvent:
        print("📊 שולף נתונים מ-JSON המובנה...")
        if not os.path.exists("project_data.json"):
            return GenerationEvent(context_str="Error: project_data.json missing.", query=ev.query)
            
        with open("project_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        context = json.dumps(data, indent=2, ensure_ascii=False)
        return GenerationEvent(context_str=context, query=ev.query)

    @step
    async def retrieve_semantic(self, ev: RetrievalEvent) -> GenerationEvent:
        print("📂 מבצע חיפוש סמנטי במסמכים...")
        if not os.path.exists("./storage"):
            return GenerationEvent(context_str="Error: Index missing.", query=ev.query)
            
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(ev.query)
        
        context = "\n\n".join([n.get_content() for n in nodes])
        return GenerationEvent(context_str=context, query=ev.query)

    @step
    async def generate_response(self, ev: GenerationEvent) -> StopEvent:
        print("🧠 מנסח תשובה סופית...")
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant. Use the context to answer clearly."),
            ChatMessage(role="user", content=f"Context: {ev.context_str}\n\nQuestion: {ev.query}")
        ]
        response = await Settings.llm.achat(messages)
        return StopEvent(result=str(response.message.content))

# --- 3. הגדרות גלובליות ---
def setup_settings():
    api_key = os.getenv("COHERE_API_KEY")
    Settings.embed_model = CohereEmbedding(model_name="embed-multilingual-v3.0", api_key=api_key)
    Settings.llm = Cohere(model="command-a-03-2025", api_key=api_key)

setup_settings()