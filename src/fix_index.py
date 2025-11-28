import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))

def create_index():
    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=MEMGRAPH_AUTH)
    with driver.session() as session:
        print("🔧 Настройка векторного индекса (metric='cos')...")
        
        try:
            session.run("DROP VECTOR INDEX chunk_vector_index")
            print("🗑️ Старый индекс удален.")
        except Exception:
            pass

        try:
            # ИСПРАВЛЕНО: "metric": "cos" (вместо "cosine")
            session.run("""
            CREATE VECTOR INDEX chunk_vector_index ON :Chunk(embedding) 
            WITH CONFIG {"dimension": 768, "metric": "cos"}
            """)
            print("✅ Векторный индекс успешно создан!")
        except Exception as e:
            print(f"❌ Ошибка создания индекса: {e}")

    driver.close()

if __name__ == "__main__":
    create_index()
