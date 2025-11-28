import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))

def inspect():
    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=MEMGRAPH_AUTH)
    with driver.session() as session:
        # Ищем документ, в названии которого есть 'finice'
        query = """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE d.id CONTAINS 'finice'
        RETURN d.id as doc, c.text as text
        LIMIT 1
        """
        result = list(session.run(query))
        
        if not result:
            print("❌ В базе НЕТ чанков для файла finice.pdf")
        else:
            doc_id = result[0]['doc']
            text = result[0]['text']
            print(f"✅ Документ найден: {doc_id}")
            print(f"📄 Длина текста: {len(text)} символов")
            print("-" * 40)
            print(f"🔍 Начало текста:\n{text[:500]}...")
            print("-" * 40)

    driver.close()

if __name__ == "__main__":
    inspect()
