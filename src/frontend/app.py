import streamlit as st
import requests
import os
from streamlit_agraph import agraph, Node, Edge, Config

# Config
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Memgraph + Gemini RAG", layout="wide")

st.title("🧠 Memgraph + Gemini RAG System")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "File Management", "Graph Visualization"])

# --- HELPERS ---
def get_files():
    try:
        res = requests.get(f"{API_URL}/files")
        if res.status_code == 200:
            return res.json().get("files", [])
    except:
        st.error("Could not connect to backend.")
    return []

# --- PAGES ---

if page == "File Management":
    st.header("📂 File Management")

    # Upload
    st.subheader("Upload New File")
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT, MD)", type=['pdf', 'txt', 'md'])
    if uploaded_file is not None:
        if st.button("Upload"):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            res = requests.post(f"{API_URL}/upload", files=files)
            if res.status_code == 200:
                st.success(f"Uploaded {uploaded_file.name}")
                st.rerun()
            else:
                st.error("Upload failed")

    # List
    st.subheader("Existing Files")
    files = get_files()
    if files:
        for f in files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 {f}")
            with col2:
                if st.button("Process", key=f"proc_{f}"):
                    res = requests.post(f"{API_URL}/process", params={"filename": f})
                    if res.status_code == 200:
                        st.info(f"Processing started for {f}")
                    else:
                        st.error("Failed to start processing")
            with col3:
                if st.button("Delete", key=f"del_{f}"):
                    res = requests.delete(f"{API_URL}/files/{f}")
                    if res.status_code == 200:
                        st.success(f"Deleted {f}")
                        st.rerun()
    else:
        st.info("No files found.")

elif page == "Chat":
    st.header("💬 Knowledge Chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call API
        try:
            res = requests.post(f"{API_URL}/query", json={"question": prompt})
            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No answer generated.")
                sources = data.get("sources", [])

                response_text = f"{answer}\n\n"
                if sources:
                    response_text += "**Sources:**\n"
                    for s in sources:
                        response_text += f"- *{s['text'][:100]}...* (Score: {s['score']:.2f})\n"

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response_text)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.error("Error getting response from backend.")
        except Exception as e:
            st.error(f"Connection error: {e}")

elif page == "Graph Visualization":
    st.header("🕸 Knowledge Graph")

    limit = st.slider("Limit nodes", 10, 500, 100)

    if st.button("Load Graph"):
        try:
            res = requests.get(f"{API_URL}/graph", params={"limit": limit})
            if res.status_code == 200:
                data = res.json()
                nodes_data = data.get("nodes", [])
                edges_data = data.get("edges", [])

                if not nodes_data:
                    st.warning("Graph is empty.")
                else:
                    nodes = []
                    for n in nodes_data:
                        nodes.append(Node(id=n["id"], label=n["id"], size=20, color="#FF6F61"))

                    edges = []
                    for e in edges_data:
                        edges.append(Edge(source=e["source"], target=e["target"], label=e["label"]))

                    config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")

                    agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.error("Failed to fetch graph data")
        except Exception as e:
            st.error(f"Error: {e}")
