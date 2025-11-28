import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))

def reset_db():
    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=MEMGRAPH_AUTH)
    with driver.session() as session:
        print("🔍 Проверка соединения...")
        try:
            ver = session.run("SHOW VERSION").single()
            print(f"ℹ️ Memgraph активен. Версия: {ver[0] if ver else 'Unknown'}")
        except Exception as e:
            print(f"⚠️ Не удалось узнать версию (но продолжаем): {e}")

        print("🧹 Полная очистка базы...")
        # Удаляем данные
        session.run("MATCH (n) DETACH DELETE n")
        
        # Удаляем индекс (без IF EXISTS, так как наша версия его не поддерживает)
        try:
            session.run("DROP VECTOR INDEX chunk_vector_index")
            print("🗑️ Старый индекс удален.")
        except Exception:
            pass # Игнорируем ошибку, если индекса не было

        print("🔧 Создание векторного индекса...")
        try:
            # Добавил 'capacity': 10000 - это помогает выделить память заранее
            session.run("""
            CREATE VECTOR INDEX chunk_vector_index ON :Chunk(embedding) 
            WITH CONFIG {"dimension": 768, "metric": "cos", "capacity": 10000}
            """)
            print("✅ Векторный индекс УСПЕШНО создан!")
            
            # Обычные индексы
            session.run("CREATE INDEX ON :Entity(id);")
            session.run("CREATE INDEX ON :Document(id);")
            print("✅ Обычные индексы созданы.")
            
        except Exception as e:
            print(f"❌ Ошибка создания индекса: {e}")
            print("💡 Если это 'Unknown exception', обязательно выполни 'docker-compose restart memgraph'")

    driver.close()

if __name__ == "__main__":
    reset_db()
