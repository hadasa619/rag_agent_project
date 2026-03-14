import os
import json
import ssl
import urllib3
from dotenv import load_dotenv

# --- הגדרות אבטחה וסינון (NetFree) ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.cohere import Cohere
from schema import ProjectKnowledgeBase  # ייבוא הסכימה שהגדרנו

# טעינת משתני סביבה
load_dotenv()

def run_extraction():
    """
    פונקציה הסורקת את התיעוד ומחלצת נתונים למבנה JSON מובנה
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ שגיאה: COHERE_API_KEY לא נמצא ב-env.")
        return

    # 1. הגדרת המודל - משתמשים במודל החזק ביותר לחילוץ (Command-A)
    llm = Cohere(model="command-a-03-2025", api_key=api_key)

    # 2. טעינת המסמכים מתיקיית ה-data
    print("📂 טוען מסמכים מתיקיית data...")
    if not os.path.exists("./data") or not os.listdir("./data"):
        print("❌ שגיאה: תיקיית data ריקה או לא קיימת.")
        return

    reader = SimpleDirectoryReader("./data")
    documents = reader.load_data()
    
    # איחוד כל הטקסט למחרוזת אחת לצורך ניתוח כולל
    full_text = "\n\n".join([d.get_content() for d in documents])

    # 3. יצירת Structured LLM
    # פונקציית העזר הזו גורמת ל-LLM להחזיר אובייקט Pydantic במקום טקסט חופשי
    structured_llm = llm.as_structured_llm(ProjectKnowledgeBase)

    print("⏳ מנתח ומחלץ נתונים מובנים (זה עשוי לקחת כמה שניות)...")
    
    try:
        # שליחת הטקסט לחילוץ
        response = structured_llm.complete(
            f"עבור על התיעוד הטכני הבא וחלץ מתוכו את כל ההחלטות, הכללים והאזהרות בצורה מדויקת:\n\n{full_text}"
        )

        # 4. המרה ל-JSON ושמירה
        # response.obj מכיל את האובייקט כפי שהוגדר ב-Pydantic
        output_data = response.obj.dict()
        
        # הוספת מטmetadata בסיסי
        final_output = {
            "schema_version": "1.0",
            "generated_at": "2026-03-15T00:00:00Z", # תאריך נוכחי
            "project_data": output_data
        }

        with open("project_data.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print("✨ הצלחה! הקובץ 'project_data.json' נוצר עם הנתונים המובנים.")
        
        # הצגת סיכום קצר של מה שחולץ
        print(f"📊 חולצו: {len(output_data['decisions'])} החלטות, "
              f"{len(output_data['rules'])} כללים, "
              f"{len(output_data['warnings'])} אזהרות.")

    except Exception as e:
        print(f"❌ חלה שגיאה בתהליך החילוץ: {e}")

if __name__ == "__main__":
    run_extraction()