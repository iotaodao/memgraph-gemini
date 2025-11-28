import os
import json
import uuid
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
import google.generativeai as genai

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def diagnose():
    print("🚀 ЗАПУСК ДИАГНОСТИКИ...")
    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=MEMGRAPH_AUTH)
    
    with driver.session() as session:
        # 1. ПРОВЕРКА ДАННЫХ
        count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        print(f"📊 Текущее количество узлов в базе: {count}")
        
        # 2. ЕСЛИ БАЗА ПУСТА - ЗАГРУЖАЕМ ТЕСТ
        if count == 0:
            print("\n⚠️ База пуста! Принудительная загрузка тестовых данных...")
            
            # Текст для загрузки
            text = "Memgraph is a graph database supported by Gemini AI."
            print(f"📝 Текст: {text}")
            
            # Эмбеддинг
            vector = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )['embedding']
            
            # Запрос на вставку (Хардкод для надежности)
            chunk_id = str(uuid.uuid4())
            try:
                # Вставляем Документ и Чанк
                q = f"""
                MERGE (d:Document {{id: 'test_doc'}})
                MERGE (c:Chunk {{id: '{chunk_id}'}})
                SET c.text = {json.dumps(text)}
                SET c.embedding = {json.dumps(vector)}
                MERGE (d)-[:HAS_CHUNK]->(c)
                """
                session.run(q)
                
                # Вставляем Сущность (чтобы граф был не пустой)
                session.run(f"""
                MATCH (c:Chunk {{id: '{chunk_id}'}})
                MERGE (e:Entity {{id: 'Memgraph'}})
                MERGE (c)-[:MENTIONS]->(e)
                """)
                print("✅ Тестовые данные успешно записаны!")
            except Exception as e:
                print(f"❌ Ошибка записи: {e}")
                return

        # 3. ПРОВЕРКА ПОИСКА (RAG)
        print("\n🔎 Тест поиска: 'What is Memgraph?'")
        q_vector = genai.embed_content(
            model="models/text-embedding-004",
            content="What is Memgraph?",
            task_type="retrieval_query"
        )['embedding']
        
        # Пробуем разные сигнатуры vector_search, так как версии меняются
        search_queries = [
            # Вариант 1 (Memgraph 2.15+ native): index, limit, vector
            f"CALL vector_search.search('chunk_vector_index', 5, {json.dumps(q_vector)}) YIELD node, score RETURN node.text, score",
            # Вариант 2 (Старый native): index, vector, limit
            f"CALL vector_search.search('chunk_vector_index', {json.dumps(q_vector)}, 5) YIELD node, score RETURN node.text, score"
        ]
        
        success = False
        for i, q in enumerate(search_queries):
            try:
                print(f"   👉 Попытка метода поиска #{i+1}...")
                res = list(session.run(q))
                if res:
                    print(f"   ✅ НАЙДЕНО: {res[0]['node.text']} (Score: {res[0]['score']:.4f})")
                    success = True
                    break
                else:
                    print("   ⚠️ Поиск отработал без ошибок, но вернул 0 результатов (низкий скор?).")
            except Exception as e:
                print(f"   ❌ Метод #{i+1} не подошел: {e}")
        
        if not success:
            print("\n💡 СОВЕТ: Если методы не подошли, возможно индекс 'chunk_vector_index' сломан.")
            print("   Попробуйте выполнить: python src/reset_and_init.py")

    driver.close()

if __name__ == "__main__":
    diagnose()
