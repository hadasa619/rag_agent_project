import os
import json
import ssl
import urllib3
import cohere
from dotenv import load_dotenv

# --- הגדרות אבטחה וסינון (NetFree) ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from llama_index.core import SimpleDirectoryReader

# טעינת משתני סביבה
load_dotenv()

def run_extraction():
    """
    סורק את תיקיית הנתונים ומחלץ מידע מובנה באמצעות ה-Chat API הישיר של Cohere.
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ שגיאה: COHERE_API_KEY לא נמצא ב-env.")
        return

    # 1. איתור נתיב תיקיית הנתונים
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "data")
    
    print(f"📂 מתחיל סריקה בנתיב: {data_path}")
    
    # 2. טעינת המסמכים (שימוש ב-Reader רק לצורך קריאת הטקסט)
    try:
        reader = SimpleDirectoryReader(input_dir=data_path, recursive=True)
        documents = reader.load_data()
        if not documents:
            print(f"❌ לא נמצאו קבצים בתוך {data_path}")
            return
        full_text = "\n\n".join([d.get_content() for d in documents])
        print(f"📄 נטענו {len(documents)} מסמכים.")
    except Exception as e:
        print(f"❌ שגיאה בטעינת הקבצים: {e}")
        return

    # 3. הגדרת ה-Client הישיר של Cohere (עוקף את LlamaIndex למניעת 404)
    co = cohere.ClientV2(api_key=api_key)

    prompt = f"""
    עבור על התיעוד הטכני הבא וחלץ מתוכו נתונים במבנה JSON מדויק.
    המבנה חייב לכלול שלוש רשימות:
    1. 'decisions': החלטות טכניות (title, summary, tags, date).
    2. 'rules': כללי UI/UX או לוקליזציה (rule, scope, notes).
    3. 'warnings': אזהרות קריטיות (area, message, severity).

    החזר אך ורק את ה-JSON, ללא הסברים וללא תגיות Markdown.

    הטקסט לניתוח:
    {full_text}
    """

    print("⏳ שולח ל-Cohere Chat API (ישיר)...")
    
    try:
        # שימוש ב-Chat API החדש (זה לעולם לא יחזיר 404 על Generate)
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # שליפת הטקסט מהתשובה
        raw_text = response.message.content[0].text
        
        # ניקוי שאריות אם קיימות
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        
        # המרה ל-JSON
        extracted_data = json.loads(clean_json)

        final_output = {
            "schema_version": "1.0",
            "generated_at": "2026-03-15",
            "project_data": extracted_data
        }

        # שמירה לקובץ
        output_file = "project_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print(f"\n✨ הצלחה! הקובץ '{output_file}' נוצר.")
        
        # הדפסת סיכום
        d = extracted_data
        print(f"✅ חולצו {len(d.get('decisions', []))} החלטות.")
        print(f"✅ חולצו {len(d.get('rules', []))} כללי עיצוב.")
        print(f"✅ חולצו {len(d.get('warnings', []))} אזהרות.")

    except Exception as e:
        print(f"❌ חלה שגיאה בעיבוד התוצאה: {e}")

if __name__ == "__main__":
    run_extraction()