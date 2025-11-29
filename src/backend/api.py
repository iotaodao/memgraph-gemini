import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from src.backend.etl import GraphPipeline
from src.backend.rag import GraphRAG
from src.backend.database import MemgraphDriver

app = FastAPI()

# Config
DATA_DIR = os.getenv("DATA_DIR", "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    error: Optional[str] = None

class GraphData(BaseModel):
    nodes: List[dict]
    edges: List[dict]

# Startup Event for DB Initialization
@app.on_event("startup")
async def startup_event():
    try:
        driver = MemgraphDriver().driver
        with driver.session() as session:
            # Create Vector Index if it doesn't exist
            # Note: MAGE vector_search.create_index typically throws if it exists, so we might need to check first or try/catch.
            # A simple check is to run it and ignore specific error "Index with name ... already exists".
            try:
                # 768 is the dimension for text-embedding-004
                session.run("CALL vector_search.create_index('chunk_vector_index', 'embedding', 'cosine', 768)")
                print("✅ Vector index 'chunk_vector_index' created.")
            except Exception as e:
                # Naive check: if it's "already exists", we are fine.
                if "already exists" in str(e):
                    print("ℹ️ Vector index already exists.")
                else:
                    print(f"⚠️ Failed to create vector index: {e}")

            # Create other constraints/indexes
            try:
                session.run("CREATE INDEX ON :Document(id);")
                session.run("CREATE INDEX ON :Chunk(id);")
                session.run("CREATE INDEX ON :Entity(id);")
                print("✅ Standard indexes created.")
            except Exception as e:
                print(f"⚠️ Index creation warning: {e}")

    except Exception as e:
        print(f"❌ Startup DB error: {e}")


# Helpers
def process_file_task(filepath: str):
    """Background task to run the ETL pipeline."""
    try:
        pipeline = GraphPipeline()
        # Drain the generator
        for status in pipeline.process_file(filepath, yield_status=True):
            print(f"[Task] {status}")
    except Exception as e:
        print(f"[Task] Error processing {filepath}: {e}")

# Endpoints

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Prevent path traversal
        filename = os.path.basename(file.filename)
        file_location = os.path.join(DATA_DIR, filename)

        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"filename": filename, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")

@app.get("/files")
async def list_files():
    files = []
    for f in os.listdir(DATA_DIR):
        if os.path.isfile(os.path.join(DATA_DIR, f)):
            files.append(f)
    return {"files": files}

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    # Prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(DATA_DIR, safe_filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"File {safe_filename} deleted"}
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/process")
async def process_file_endpoint(filename: str, background_tasks: BackgroundTasks):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(DATA_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    background_tasks.add_task(process_file_task, file_path)
    return {"message": f"Processing of {safe_filename} started in background"}

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(request: QueryRequest):
    rag = GraphRAG()
    result = rag.search(request.question)
    if "error" in result:
        return QueryResponse(answer="", sources=[], error=result["error"])
    return QueryResponse(answer=result["answer"], sources=result["sources"])

@app.get("/graph", response_model=GraphData)
async def get_graph_data(limit: int = 100):
    driver = MemgraphDriver().driver
    nodes = []
    edges = []

    with driver.session() as session:
        # Fetch Entities
        q_nodes = "MATCH (n:Entity) RETURN n.id as id, n.type as type LIMIT $limit"
        res_nodes = session.run(q_nodes, limit=limit)
        for r in res_nodes:
            nodes.append({"id": r["id"], "label": r["id"], "type": r["type"]})

        # Fetch Relationships
        q_rels = "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.id as source, b.id as target, type(r) as type LIMIT $limit"
        res_rels = session.run(q_rels, limit=limit)
        for r in res_rels:
            edges.append({"source": r["source"], "target": r["target"], "label": r["type"]})

    return GraphData(nodes=nodes, edges=edges)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
