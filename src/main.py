print("🚀 [1/6] Инициализация Python...")
import os
import json
import uuid
import time
import re
import glob
import sys
# ДОБАВЛЕНО: Импорт типов
from typing import List, Dict, Any

# Ловим ошибки импорта библиотек
try:
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai
    from chonkie import TokenChunker
    from neo4j import GraphDatabase
    print("✅ [2/6] Библиотеки загружены")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

# --- КОНФИГУРАЦИЯ ---
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))

if "GEMINI_API_KEY" not in os.environ:
    print("❌ Ошибка: Не найден GEMINI_API_KEY")
    sys.exit(1)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class HybridGraphPipeline:
    def __init__(self, uri, auth, extraction_model="gemini-2.5-flash"):
        print(f"🔌 [3/6] Подключение к Memgraph ({uri})...")
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            self.driver.verify_connectivity()
            print("✅ Подключение успешно!")
        except Exception as e:
            print(f"❌ Не удалось подключиться к базе: {e}")
            sys.exit(1)
            
        self.chunker = TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=50)
        self.extraction_model = genai.GenerativeModel(
            model_name=extraction_model,
            system_instruction="Extract entities (Person, Org, Tech) and relationships (SCREAMING_SNAKE_CASE). JSON output: {entities: [], relations: []}.",
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        self.embedding_model_name = "models/text-embedding-004" 

    def close(self):
        self.driver.close()

    def _generate_embedding(self, text: str) -> List[float]:
        try:
            return genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )['embedding']
        except Exception as e:
            print(f"⚠️ Ошибка вектора: {e}")
            return []

    def _extract_graph_data(self, text: str) -> Dict[str, Any]:
        try:
            resp = self.extraction_model.generate_content(f"Extract graph from:\\n\\n{text}")
            return json.loads(resp.text)
        except:
            return {"entities": [], "relations": []}

    def process_directory(self, data_dir: str):
        abs_path = os.path.abspath(data_dir)
        print(f"📂 [4/6] Сканирование папки: {abs_path}")
        
        if not os.path.exists(data_dir):
            print(f"❌ Папка '{data_dir}' не существует!")
            return

        files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True)
        
        print(f"📄 Найдено файлов: {len(files)}")
        if len(files) == 0:
            print("⚠️ Папка пуста или файлы не имеют расширения .txt/.md")
            return
        
        print("▶️ [5/6] Начало обработки...")
        with self.driver.session() as session:
            for filepath in files:
                filename = os.path.basename(filepath)
                # Нормализация ID
                doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
                print(f"   🔪 Читаю файл: {filename}")
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f: 
                        text = f.read().replace('\\0', '')
                    
                    if not text.strip():
                        print("   ⚠️ Файл пуст, пропускаю.")
                        continue

                    chunks = self.chunker(text)
                    print(f"      🧩 Чанков: {len(chunks)}")

                    # 1. Документ (JSON DUMPS)
                    session.run(f"MERGE (d:Document {{id: {json.dumps(doc_id)}}})")

                    for i, chunk in enumerate(chunks):
                        graph_data = self._extract_graph_data(chunk.text)
                        vector = self._generate_embedding(chunk.text)
                        chunk_id = str(uuid.uuid4())

                        # 2. Чанк (JSON DUMPS)
                        query_chunk = f"""
                        MATCH (d:Document {{id: {json.dumps(doc_id)}}})
                        MERGE (c:Chunk {{id: {json.dumps(chunk_id)}}})
                        SET c.index = {i}
                        SET c.text = {json.dumps(chunk.text)}
                        SET c.embedding = {json.dumps(vector)}
                        MERGE (d)-[:HAS_CHUNK]->(c)
                        """
                        session.run(query_chunk)

                        # 3. Сущности
                        for ent in graph_data.get("entities", []):
                            e_id = ent.get("id", "").strip()
                            e_type = ent.get("type", "Thing").strip()
                            if not e_id: continue
                            
                            e_type = re.sub(r'[^a-zA-Z0-9_]', '', e_type) or "Thing"
                            
                            q_ent = f"""
                            MATCH (c:Chunk {{id: {json.dumps(chunk_id)}}}) 
                            MERGE (e:Entity {{id: {json.dumps(e_id)}}}) 
                            ON CREATE SET e.type = {json.dumps(e_type)}
                            MERGE (c)-[:MENTIONS]->(e)
                            """
                            session.run(q_ent)

                        # 4. Связи
                        for rel in graph_data.get("relations", []):
                            src = rel.get("source", "").strip()
                            tgt = rel.get("target", "").strip()
                            if not src or not tgt: continue
                            
                            r_type = re.sub(r'[^a-zA-Z0-9_]', '', rel.get("type", "RELATED").replace(" ", "_").upper()) 
                            if not r_type: r_type = "RELATED"
                            
                            q_rel = f"""
                            MATCH (a:Entity {{id: {json.dumps(src)}}}), (b:Entity {{id: {json.dumps(tgt)}}}) 
                            MERGE (a)-[:{r_type}]->(b)
                            """
                            session.run(q_rel)
                    
                    print(f"      ✅ Файл {filename} загружен в граф.")
                            
                except Exception as e:
                    print(f"      ❌ Ошибка обработки файла: {e}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
        print("⚠️ Папка data не найдена, создана пустая.")
    
    try:
        pipeline = HybridGraphPipeline(MEMGRAPH_URI, MEMGRAPH_AUTH)
        pipeline.process_directory("data")
        pipeline.close()
        print("🎉 [6/6] Все задачи выполнены.")
    except Exception as e:
        print(f"\n❌ Критический сбой: {e}")
