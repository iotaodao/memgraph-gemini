import os
import json
import uuid
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from chonkie import TokenChunker
from neo4j import GraphDatabase

# --- КОНФИГУРАЦИЯ ---
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("⚠️ Ошибка: Не найден GEMINI_API_KEY в .env или переменных среды")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_INSTRUCTION = """
You are an expert Knowledge Graph Engineer.
Extract structured knowledge from the provided text chunk.

RULES:
1. Extract key **Entities** (Person, Org, Tech, Concept, Location). Normalize IDs.
2. Extract **Relationships**. Use SCREAMING_SNAKE_CASE.
3. Output MUST be valid JSON: {"entities": [], "relations": []}.
"""

class HybridGraphPipeline:
    def __init__(self, uri, auth, extraction_model="gemini-1.5-flash"):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.chunker = TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=50)
        self.extraction_model = genai.GenerativeModel(
            model_name=extraction_model,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        self.embedding_model_name = "models/text-embedding-004" 
        self._setup_database()
        print(f"✅ Пайплайн готов. Graph: {extraction_model}")

    def close(self):
        self.driver.close()

    def _setup_database(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX ON :Entity(id);")
            session.run("CREATE INDEX ON :Document(id);")
            try:
                session.run("CREATE VECTOR INDEX chunk_vector_index ON :Chunk(embedding) DIMENSIONS 768 METRIC COSINE;")
            except Exception:
                pass

    def _generate_embedding(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document",
                title="Chunk" 
            )
            return result['embedding']
        except Exception as e:
            print(f"⚠️ Ошибка эмбеддинга: {e}")
            return []

    def _extract_graph_data(self, text: str) -> Dict[str, Any]:
        try:
            response = self.extraction_model.generate_content(
                f"Extract knowledge from this text:\\n\\n{text}",
                safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
            )
            data = json.loads(response.text)
            if "entities" not in data: data["entities"] = []
            if "relations" not in data: data["relations"] = []
            return data
        except Exception as e:
            print(f"⚠️ Ошибка экстракции: {e}")
            return {"entities": [], "relations": []}

    def process_document(self, doc_id: str, full_text: str):
        print(f"🔪 Нарезка документа '{doc_id}'...")
        chunks = self.chunker(full_text)
        
        with self.driver.session() as session:
            session.run("MERGE (d:Document {id: $doc_id})", doc_id=doc_id)

            for i, chunk in enumerate(chunks):
                chunk_text = chunk.text
                chunk_uuid = str(uuid.uuid4())
                print(f"   ⚙️ Чанк {i+1}/{len(chunks)}...")
                
                graph_data = self._extract_graph_data(chunk_text)
                vector_data = self._generate_embedding(chunk_text)
                
                session.write_transaction(
                    self._ingest_chunk,
                    doc_id, i, chunk_uuid, chunk_text,
                    graph_data, vector_data
                )
                time.sleep(0.5)

        print(f"✅ Документ '{doc_id}' обработан.")

    @staticmethod
    def _ingest_chunk(tx, doc_id, index, chunk_id, text, graph_data, vector):
        query_chunk = """
        MATCH (d:Document {id: $doc_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.index = $index, c.text = $text, c.embedding = $vector 
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        tx.run(query_chunk, doc_id=doc_id, chunk_id=chunk_id, index=index, text=text, vector=vector)

        for entity in graph_data["entities"]:
            query_entity = """
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (e:Entity {id: $e_id})
            ON CREATE SET e.type = $e_type
            MERGE (c)-[:MENTIONS]->(e)
            """
            tx.run(query_entity, chunk_id=chunk_id, e_id=entity["id"], e_type=entity["type"])

        for rel in graph_data["relations"]:
            rel_type = rel["type"].replace(" ", "_").upper()
            query_rel = f"""
            MATCH (a:Entity {{id: $source}}), (b:Entity {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            """
            tx.run(query_rel, source=rel["source"], target=rel["target"])

if __name__ == "__main__":
    text = """
    Memgraph is a high-performance graph database written in C++. 
    It supports vector search using cosine similarity, making it ideal for AI apps.
    By using Google Gemini embeddings (model text-embedding-004), developers can build Hybrid RAG systems.
    """
    try:
        pipeline = HybridGraphPipeline(MEMGRAPH_URI, MEMGRAPH_AUTH)
        pipeline.process_document("doc_hybrid_v1", text)
        pipeline.close()
    except Exception as e:
        print(f"\\n❌ Ошибка: {e}")
