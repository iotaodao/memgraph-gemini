import os
import json
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from neo4j import GraphDatabase

# --- НАСТРОЙКИ ---
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))
EMBEDDING_MODEL = "models/text-embedding-004"
QA_MODEL = "gemini-2.5-flash"

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("⚠️ Ошибка: Не найден GEMINI_API_KEY в .env")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_embedding(text):
    return genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query" 
    )['embedding']

def generate_answer(question, context):
    model = genai.GenerativeModel(QA_MODEL)
    prompt = f"""
    You are a helpful assistant. Answer the question based strictly on the Context provided.
    
    Context:
    {context}
    
    Question: {question}
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

def search(question):
    print(f"\n🔎 Вопрос: {question}")
    
    try:
        vector = get_embedding(question)
    except Exception as e:
        print(f"❌ Ошибка создания эмбеддинга: {e}")
        return

    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=MEMGRAPH_AUTH)
    
    with driver.session() as session:
        # Вектор конвертируем в строку JSON
        vec_str = json.dumps(vector)
        
        # Запрос с OPTIONAL MATCH для защиты от отсутствующих связей
        query = f"""
        CALL vector_search.search('chunk_vector_index', 3, {vec_str}) 
        YIELD node, score
        OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
        RETURN node.text as text, score, collect(e.id) as entities
        """
        
        try:
            result = session.run(query)
            records = list(result)
        except Exception as e:
            print(f"❌ Ошибка Memgraph: {e}")
            return
        
        if not records:
            print("⚠️ Ничего не найдено.")
            return

        print(f"✅ Найдено источников: {len(records)}\n")
        
        context_text = ""
        for i, r in enumerate(records):
            # --- ИСПРАВЛЕНИЕ ОШИБКИ SCORE ---
            raw_score = r.get('score')
            if raw_score is None:
                score_display = "N/A"
            else:
                try:
                    score_display = f"{float(raw_score):.4f}"
                except:
                    score_display = str(raw_score)
            # --------------------------------
            
            text = r.get('text', "")
            # Обработка списка сущностей (может быть [None] из-за OPTIONAL MATCH)
            ent_list = r.get('entities', [])
            valid_entities = [str(e) for e in ent_list if e is not None]
            entities_str = ', '.join(valid_entities) if valid_entities else "(Нет связей)"
            
            print(f"--- Источник {i+1} (Score: {score_display}) ---")
            print(f"🔗 Сущности: {entities_str}")
            print(f"📄 Текст: {text[:100].replace(chr(10), ' ')}...")
            
            context_text += f"Source {i+1}:\nText: {text}\nEntities: {entities_str}\n\n"

        print("\n🧠 Генерирую ответ...")
        try:
            answer = generate_answer(question, context_text)
            print("\n" + "="*20 + " ОТВЕТ " + "="*20)
            print(answer)
            print("="*47)
        except Exception as e:
            print(f"❌ Ошибка генерации ответа: {e}")

    driver.close()

if __name__ == "__main__":
    search("What is Memgraph?")
