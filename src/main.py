print("🚀 [1/6] Инициализация Python...")
import os
import json
import uuid
import time
import re
import glob
import sys
from typing import List, Dict, Any

# Ловим ошибки импорта
try:
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai
    from chonkie import TokenChunker
    from neo4j import GraphDatabase
    # --- НОВОЕ: Импорт Docling для PDF ---
    from docling.document_converter import DocumentConverter
    print("✅ [2/6] Библиотеки загружены (включая Docling)")
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

# Системный промпт
SYSTEM_PROMPT = """
You are an expert Knowledge Graph Engineer.
Extract entities and relationships from the text.

STRICT JSON OUTPUT FORMAT (NO MARKDOWN, NO COMMENTS):
{
  "entities": [
    {"id": "Entity Name", "type": "Category"}
  ],
  "relations": [
    {"source": "Entity Name", "target": "Entity Name", "type": "RELATION_TYPE"}
  ]
}
Normalize IDs. Use SCREAMING_SNAKE_CASE for relation types.
"""

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
        # Инициализация конвертера PDF
        self.pdf_converter = DocumentConverter()
        
        self.extraction_model = genai.GenerativeModel(
            model_name=extraction_model,
            system_instruction=SYSTEM_PROMPT,
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
            raw = resp.text
            if "```" in raw:
                raw = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw)
        except:
            return {"entities": [], "relations": []}

    def _read_file_content(self, filepath: str) -> str:
        """Читает файл в зависимости от расширения."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".pdf":
            try:
                # Docling конвертирует PDF в Markdown
                result = self.pdf_converter.convert(filepath)
                return result.document.export_to_markdown()
            except Exception as e:
                print(f"      ❌ Ошибка Docling: {e}")
                return ""
        else:
            # Обычный текст
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().replace('\\0', '')

    def process_directory(self, data_dir: str):
        abs_path = os.path.abspath(data_dir)
        print(f"📂 [4/6] Сканирование папки: {abs_path}")
        
        if not os.path.exists(data_dir):
            print(f"❌ Папка '{data_dir}' не существует!")
            return

        # Добавили поиск .pdf
        files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
        
        print(f"📄 Найдено файлов: {len(files)}")
        if not files: return
        
        print("▶️ [5/6] Начало обработки...")
        with self.driver.session() as session:
            for filepath in files:
                filename = os.path.basename(filepath)
                doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
                print(f"   🔪 Читаю файл: {filename}")
                
                try:
                    # Универсальное чтение
                    text = self._read_file_content(filepath)
                    
                    if not text.strip():
                        print("   ⚠️ Файл пуст или не прочитан.")
                        continue

                    chunks = self.chunker(text)
                    print(f"      🧩 Чанков: {len(chunks)}")

                    # 1. Документ
                    session.run(f"MERGE (d:Document {{id: {json.dumps(doc_id)}}})")

                    for i, chunk in enumerate(chunks):
                        graph_data = self._extract_graph_data(chunk.text)
                        vector = self._generate_embedding(chunk.text)
                        chunk_id = str(uuid.uuid4())

                        # 2. Чанк
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
                            e_id = ent.get("id") or ent.get("name")
                            if not e_id: continue
                            e_id = e_id.strip()
                            e_type = re.sub(r'[^a-zA-Z0-9_]', '', ent.get("type", "Thing").strip()) or "Thing"
                            
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
                            
                            r_type = re.sub(r'[^a-zA-Z0-9_]', '', rel.get("type", "RELATED").replace(" ", "_").upper()) or "RELATED"
                            
                            q_rel = f"""
                            MATCH (a:Entity {{id: {json.dumps(src)}}}), (b:Entity {{id: {json.dumps(tgt)}}}) 
                            MERGE (a)-[:{r_type}]->(b)
                            """
                            session.run(q_rel)
                    
                    print(f"      ✅ Файл {filename} загружен.")
                            
                except Exception as e:
                    print(f"      ❌ Ошибка: {e}")

if __name__ == "__main__":
    if not os.path.exists("data"): os.makedirs("data")
    try:
        pipeline = HybridGraphPipeline(MEMGRAPH_URI, MEMGRAPH_AUTH)
        pipeline.process_directory("data")
        pipeline.close()
        print("🎉 [6/6] Готово.")
    except Exception as e:
        print(f"\n❌ Сбой: {e}")
