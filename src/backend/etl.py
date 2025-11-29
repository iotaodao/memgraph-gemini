import os
import json
import uuid
import re
from typing import List, Dict, Any, Generator
import google.generativeai as genai
from chonkie import TokenChunker
from docling.document_converter import DocumentConverter
from neo4j import GraphDatabase
from src.backend.database import MemgraphDriver

class GraphPipeline:
    def __init__(self, embedding_model="models/text-embedding-004", extraction_model="gemini-2.5-flash"):
        if "GEMINI_API_KEY" not in os.environ:
             raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.driver = MemgraphDriver().driver

        self.chunker = TokenChunker(tokenizer="gpt2", chunk_size=512, chunk_overlap=50)
        self.pdf_converter = DocumentConverter()

        self.extraction_system_prompt = """
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

        self.extraction_model = genai.GenerativeModel(
            model_name=extraction_model,
            system_instruction=self.extraction_system_prompt,
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        self.embedding_model_name = embedding_model

    def _generate_embedding(self, text: str) -> List[float]:
        try:
            return genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )['embedding']
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
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
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            try:
                result = self.pdf_converter.convert(filepath)
                return result.document.export_to_markdown()
            except Exception as e:
                print(f"❌ Docling error: {e}")
                raise e
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().replace('\\0', '')

    def process_file(self, filepath: str, yield_status=False) -> Generator[str, None, None]:
        filename = os.path.basename(filepath)
        doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)

        yield f"Reading file: {filename}"

        try:
            text = self._read_file_content(filepath)
            if not text.strip():
                yield "File is empty."
                return

            chunks = self.chunker(text)
            yield f"Created {len(chunks)} chunks."

            with self.driver.session() as session:
                session.run("MERGE (d:Document {id: $doc_id})", doc_id=doc_id)

                for i, chunk in enumerate(chunks):
                    if yield_status:
                        yield f"Processing chunk {i+1}/{len(chunks)}"

                    graph_data = self._extract_graph_data(chunk.text)
                    vector = self._generate_embedding(chunk.text)
                    chunk_id = str(uuid.uuid4())

                    query_chunk = """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.index = $index
                    SET c.text = $text
                    SET c.embedding = $embedding
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """
                    session.run(query_chunk, doc_id=doc_id, chunk_id=chunk_id, index=i, text=chunk.text, embedding=vector)

                    for ent in graph_data.get("entities", []):
                        e_id = ent.get("id") or ent.get("name")
                        if not e_id: continue
                        e_id = e_id.strip()
                        e_type = re.sub(r'[^a-zA-Z0-9_]', '', ent.get("type", "Thing").strip()) or "Thing"

                        q_ent = """
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (e:Entity {id: $e_id})
                        ON CREATE SET e.type = $e_type
                        MERGE (c)-[:MENTIONS]->(e)
                        """
                        session.run(q_ent, chunk_id=chunk_id, e_id=e_id, e_type=e_type)

                    for rel in graph_data.get("relations", []):
                        src = rel.get("source", "").strip()
                        tgt = rel.get("target", "").strip()
                        if not src or not tgt: continue

                        r_type = re.sub(r'[^a-zA-Z0-9_]', '', rel.get("type", "RELATED").replace(" ", "_").upper()) or "RELATED"

                        # Note: Relationship types cannot be parameterized in Cypher (e.g. [:TYPE]).
                        # We sanitize r_type above with regex to be safe.
                        q_rel = f"""
                        MATCH (a:Entity {{id: $src}}), (b:Entity {{id: $tgt}})
                        MERGE (a)-[:{r_type}]->(b)
                        """
                        session.run(q_rel, src=src, tgt=tgt)

            yield f"Successfully processed {filename}"

        except Exception as e:
            yield f"Error processing {filename}: {str(e)}"
            print(f"Error: {e}")
