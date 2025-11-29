import os
import json
import google.generativeai as genai
from src.backend.database import MemgraphDriver

class GraphRAG:
    def __init__(self, embedding_model="models/text-embedding-004", qa_model="gemini-2.5-flash"):
        if "GEMINI_API_KEY" not in os.environ:
             raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.driver = MemgraphDriver().driver
        self.embedding_model = embedding_model
        self.qa_model = qa_model

    def get_embedding(self, text):
        return genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query"
        )['embedding']

    def generate_answer(self, question, context):
        model = genai.GenerativeModel(self.qa_model)
        prompt = f"""
        You are a helpful assistant. Answer the question based strictly on the Context provided.

        Context:
        {context}

        Question: {question}
        Answer:
        """
        response = model.generate_content(prompt)
        return response.text

    def search(self, question: str):
        try:
            vector = self.get_embedding(question)
        except Exception as e:
            return {"error": f"Embedding error: {e}"}

        with self.driver.session() as session:
            # Using parameter substitution for the vector
            # Note: vector_search.search() procedure call format depends on implementation.
            # Usually parameters can be passed.

            # Memgraph's vector_search module might expect a list literal in the string for the query vector if strictly following MAGE docs,
            # but standard Cypher parameters are safer and cleaner if supported.
            # Assuming standard Cypher behavior:

            query = """
            CALL vector_search.search('chunk_vector_index', 3, $vector)
            YIELD node, score
            OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
            RETURN node.text as text, score, collect(e.id) as entities
            """

            try:
                result = session.run(query, vector=vector)
                records = list(result)
            except Exception as e:
                return {"error": f"Database error: {e}"}

            if not records:
                return {"answer": "No relevant information found.", "sources": []}

            context_text = ""
            sources = []

            for i, r in enumerate(records):
                score = r.get('score', 0.0)
                text = r.get('text', "")
                ent_list = r.get('entities', [])
                valid_entities = [str(e) for e in ent_list if e is not None]
                entities_str = ', '.join(valid_entities) if valid_entities else "(No entities)"

                sources.append({
                    "text": text,
                    "score": float(score) if score is not None else 0.0,
                    "entities": valid_entities
                })

                context_text += f"Source {i+1}:\nText: {text}\nEntities: {entities_str}\n\n"

            try:
                answer = self.generate_answer(question, context_text)
                return {"answer": answer, "sources": sources}
            except Exception as e:
                return {"error": f"Generation error: {e}"}
